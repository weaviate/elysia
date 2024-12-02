import dspy
from rich import print
from rich.panel import Panel

# Util
from elysia.util.logging import backend_print
from elysia.util.parsing import update_current_message

# Prompt Executors
from elysia.querying.prompt_executors import (
    QueryExecutor, 
    ObjectSummaryExecutor
)

# Objects
from elysia.tree.objects import (
    Returns, TreeData, ActionData, DecisionData
)
from elysia.text.objects import Response
from elysia.api.objects import (
    Status, Warning, Error, Branch, TreeUpdate
)
from elysia.querying.objects import (
    GenericRetrieval, 
    MessageRetrieval, 
    ConversationRetrieval, 
    TicketRetrieval, 
    EcommerceRetrieval
)

class AgenticQuery:

    def __init__(self, 
                 base_lm: dspy.LM,
                 complex_lm: dspy.LM,
                 query_filepath: str = "elysia/training/dspy_models/query/fewshot_k8.json", 
                 collection_names: list[str] = None, 
                 collection_return_types: dict[str, list[str]] = None,
                 verbosity: int = 0): 
        
        self.verbosity = verbosity
        self.collection_names = collection_names
        self.collection_return_types = collection_return_types

        self.base_lm = base_lm
        self.complex_lm = complex_lm

        self.querier = QueryExecutor(collection_names).activate_assertions(max_backtracks=3)
        self.object_summariser = ObjectSummaryExecutor().activate_assertions(max_backtracks=1)
        if len(query_filepath) > 0:
            self.querier.load(query_filepath)

            if self.verbosity > 0:
                backend_print(f"[green]Loaded querier[/green] model at [italic magenta]{query_filepath}[/italic magenta]")

    def set_collection_names(self, collection_names: list[str]):
        self.collection_names = collection_names
        self.querier.set_collection_names(collection_names)

    def _find_previous_queries(self, available_information: Returns):
        self.previous_queries = []
        for collection_name in self.collection_names:
            if collection_name in available_information.retrieved:
                metadata = available_information.retrieved[collection_name].metadata
                if "previous_queries" in metadata:
                    self.previous_queries.append({"collection_name": collection_name, "previous_queries": metadata["previous_queries"]})  
        
    async def __call__(
            self,
            tree_data: TreeData,
            action_data: ActionData,
            decision_data: DecisionData
        ):
        
        # Get some metadata about the collection
        self._find_previous_queries(decision_data.available_information)

        # Save some variables for use only in this function
        current_message = tree_data.current_message

        # Query the collection
        Branch({
            "name": "Query Executor",
            "description": "Write code and query the collection to retrieve objects."
        })
        yield Status(f"Writing query")

        with dspy.context(lm = self.complex_lm):

            # Run the query executor (write and execute the query)
            response, query, error_message = self.querier(
                user_prompt = tree_data.user_prompt, 
                conversation_history = tree_data.conversation_history,
                previous_queries = self.previous_queries, 
                data_queried = tree_data.data_queried_string(),
                collection_information = action_data.collection_information,
                collection_return_types = action_data.collection_return_types,
                previous_reasoning = tree_data.previous_reasoning,
                current_message = current_message
            )

        if query is None: # either an error or the query is impossible
            yield GenericRetrieval([], {"collection_name": "", "impossible_prompts": [tree_data.user_prompt]})
            if error_message != "": # an error in the prompt executor
                yield Error(error_message)
            return
        
        if self.verbosity > 0:
            print(Panel.fit(query.code, title="Query code", padding=(1,1), border_style="yellow"))
            backend_print(f"[yellow]Query output type[/yellow]: {query.output_type}")
            backend_print(f"[yellow]Query collection name[/yellow]: {query.collection_name}")
            backend_print(f"[yellow]Query return type[/yellow]: {query.return_type}")

        current_message, message_update = update_current_message(current_message, query.text_return)

        # Yield results to front end
        yield Response([{"text": message_update}], {})
        yield TreeUpdate(from_node="query", to_node="query_executor", reasoning=query.reasoning, last = query.return_type != "summary")
        yield Status(f"Retrieved {len(response.objects)} objects from {query.collection_name}")

        # Add the query code to the previous queries
        self.previous_queries.append(query.code)

        # Get the objects from the response (query executor)
        objects = []
        for obj in response.objects:
            objects.append({k: v for k, v in obj.properties.items()})
            objects[-1]["uuid"] = obj.uuid.hex

        # -- (Optional) Step 4: Summarise the objects
        if query.output_type == "summary":
            Branch({
                "name": "Object Summariser",
                "description": "Generate itemised summaries of the retrieved objects."
            })
            yield Status(f"Generating summaries of the retrieved objects")

            try:

                # Run the object summariser
                summary_list, summariser = self.object_summariser(objects, current_message)
                current_message, message_update = update_current_message(current_message, summariser.text_return)

                # Yield results to front end
                yield Response([{"text": message_update}], {})
                yield TreeUpdate(from_node="query_executor", to_node="object_summariser", reasoning=summariser.reasoning, last = True)
                yield Status(f"Summarised {len(summariser)} objects")

                # attach summaries to objects
                for i, obj in enumerate(objects):
                    if i < len(summary_list):
                        obj["summary"] = summary_list[i]
                    else:
                        obj["summary"] = ""
            
            except Exception as e:
                yield Error(f"Error in object summarisation: {e}")

        # If no summarisation, attach empty strings
        else:
            yield TreeUpdate(from_node="query_executor", to_node="object_summariser", reasoning="This step was skipped because it was determined that the output type was not a summary.", last = True)
            for obj in objects:
                obj["summary"] = ""

        metadata = {
            "previous_queries": [query.code], 
            "collection_name": query.collection_name,
            "return_type": query.return_type,
            "output_type": query.output_type,
            "last_code": {
                "language": "python",
                "title": "Query",
                "text": query.code
            }
        }

        if query.return_type == "conversation":
            yield ConversationRetrieval(objects, metadata)
        elif query.return_type == "message":
            yield MessageRetrieval(objects, metadata)
        elif query.return_type == "ticket":
            yield TicketRetrieval(objects, metadata)
        elif query.return_type == "ecommerce":
            yield EcommerceRetrieval(objects, metadata)
        else:
            yield GenericRetrieval(objects, metadata)
        
