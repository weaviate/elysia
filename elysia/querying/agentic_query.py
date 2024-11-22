import datetime
import dspy


from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference
from elysia.util.logging import backend_print
from elysia.util.parsing import update_current_message
from elysia.util.collection_metadata import get_collection_data_types

from elysia.querying.prompt_executors import (
    QueryExecutor, 
    ObjectSummaryExecutor
)
from elysia.tree.objects import Returns, Objects, Status, Warning, Error, Branch, TreeUpdate
from elysia.text.objects import Response, Code
from elysia.querying.objects import GenericRetrieval, MessageRetrieval, ConversationRetrieval, TicketRetrieval, EcommerceRetrieval

class AgenticQuery:

    def __init__(self, 
                 query_creator_filepath: str = "elysia/training/dspy_models/agentic_query/fewshot_k12.json", 
                 collection_names: list[str] = None, 
                 return_types: dict[str, str] = None): 
        
        self.collection_names = collection_names

        self.querier = QueryExecutor(collection_names, return_types).activate_assertions(max_backtracks=3)
        
        # if len(query_creator_filepath) > 0:
        #     self.querier.load(query_creator_filepath)
        #     backend_print(f"[green]Loaded querier[/green] model at [italic magenta]{query_creator_filepath}[/italic magenta]")

    def set_collection_names(self, collection_names: list[str]):
        self.collection_names = collection_names
        self.querier.available_collections = collection_names

    def _find_previous_queries(self, available_information: Returns):

        self.previous_queries = []
        for collection_name in self.collection_names:
            if collection_name in available_information.retrieved:
                metadata = available_information.retrieved[collection_name].metadata
                if "previous_queries" in metadata:
                    self.previous_queries.append({"collection_name": collection_name, "previous_queries": metadata["previous_queries"]})  
    
    def _get_collection_fields(self, collection_name: str):
        example_field = client.collections.get(collection_name).query.fetch_objects(limit=1).objects[0].properties
        for key in example_field:
            if isinstance(example_field[key], datetime.datetime):
                example_field[key] = example_field[key].isoformat()
            elif not isinstance(example_field[key], str):
                example_field[key] = str(example_field[key])

        data_types = get_collection_data_types(collection_name)

        return data_types, example_field
        
    async def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs):
        
        data_queried = kwargs.get("data_queried", [])
        collection_information = kwargs.get("collection_information", [])
        current_message = kwargs.get("current_message", "")

        # Get some metadata about the collection
        self._find_previous_queries(available_information)

        # Query the collection
        Branch({
            "name": "Query Executor",
            "description": "Write code and query the collection to retrieve objects."
        })
        yield Status(f"Writing query")

        try:
            # Run the query executor (write and execute the query)
            response, query = self.querier(
                user_prompt = user_prompt, 
                previous_queries = self.previous_queries, 
                data_queried = data_queried,
                collection_information = collection_information,
                previous_reasoning = previous_reasoning,
                current_message = current_message
            )

        except Exception as e:
            yield Error(f"Error in query execution: {e}")

        # If the query is not possible, yield a generic retrieval and return nothing
        if query is None:
            yield GenericRetrieval([], {"collection_name": query.collection_name, "impossible_prompts": [user_prompt]})
            return

        current_message, message_update = update_current_message(current_message, query.text_return)

        # Yield results to front end
        if message_update != "":
            yield Response([{"text": message_update}], {})
        yield Code([{"text": query.code, "language": "python", "title": "Query"}], {})
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
        if query.return_type == "summary":
            Branch({
                "name": "Object Summariser",
                "description": "Generate itemised summaries of the retrieved objects."
            })
            yield Status(f"Generating summaries of the retrieved objects")

            try:

                # Run the object summariser
                summary_list, summariser = self.object_summariser(objects)

                # Yield results to front end
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
        
