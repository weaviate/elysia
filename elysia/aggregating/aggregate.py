import dspy
from rich import print
from rich.panel import Panel

# Util
from elysia.util.logging import backend_print
from elysia.util.parsing import update_current_message, format_aggregation_response

# Objects
from elysia.api.objects import TreeUpdate, Status, Error, Warning, Branch
from elysia.tree.objects import TreeData, ActionData, DecisionData
from elysia.tree.objects import Returns
from elysia.text.objects import Response
from elysia.aggregating.objects import GenericAggregation

# Prompt Executors
from elysia.aggregating.prompt_executors import AggregateExecutor

class AgenticAggregate:

    def __init__(self, 
                 base_lm: dspy.LM,
                 complex_lm: dspy.LM,
                 aggregate_initialiser_filepath: str = "elysia/training/dspy_models/aggregate_initialiser/fewshot_k12.json", 
                 aggregate_executor_filepath: str = "elysia/training/dspy_models/aggregate_executor/fewshot_k12.json",
                 collection_names: list[str] = None,
                 verbosity: int = 0):
        
        self.verbosity = verbosity
        self.base_lm = base_lm
        self.complex_lm = complex_lm
        self.collection_names = collection_names
        self.aggregate_executor = AggregateExecutor(collection_names=collection_names).activate_assertions()
    
    def _find_previous_aggregations(self, available_information: Returns):

        self.previous_aggregations = []
        for collection_name in self.collection_names:
            if collection_name in available_information.aggregation:
                metadata = available_information.aggregation[collection_name].metadata
                if "previous_aggregations" in metadata:
                    self.previous_aggregations.append({"collection_name": collection_name, "previous_aggregations": metadata["previous_aggregations"]})  

    async def __call__(self, 
            tree_data: TreeData,
            action_data: ActionData,
            decision_data: DecisionData
        ):

        # set some temporary variables
        current_message = tree_data.current_message
        
        # Get some metadata about the collection
        self._find_previous_aggregations(decision_data.available_information)

        # -- Step 2: Aggregate
        Branch({
            "name": "Aggregation Executor",
            "description": "Write code and aggregate the collection."
        })
        yield Status(f"Writing aggregation")

        with dspy.context(lm = self.complex_lm):
            response, aggregation, error_message = self.aggregate_executor(
                user_prompt=tree_data.user_prompt, 
                conversation_history=tree_data.conversation_history,
                data_queried=tree_data.data_queried_string(), 
                collection_information=action_data.collection_information, 
                previous_reasoning=tree_data.previous_reasoning,
                previous_aggregations=self.previous_aggregations
            )
            
        # If there is an error, yield a generic retrieval and return nothing
        if aggregation is None:
            yield GenericAggregation([], {"collection_name": aggregation.collection_name, "impossible_prompts": [tree_data.user_prompt]})
            if error_message != "":
                yield Error(error_message)
            return

        if self.verbosity > 0:
            print(Panel.fit(aggregation.code, title="Aggregation code", padding=(1,1), border_style="yellow"))

        current_message, message_update = update_current_message(current_message, aggregation.text_return)

        # return values
        objects = [
            {aggregation.collection_name: format_aggregation_response(response)}
        ]
        metadata = {
            "collection_name": aggregation.collection_name,
            "previous_aggregations": self.previous_aggregations,
            "description": [aggregation.description],
            "last_code": {
                "language": "python",
                "title": "Aggregation",
                "text": aggregation.code
            }
        }

        # Yield results to front end
        yield Response([{"text": message_update}], {})
        yield TreeUpdate(from_node="aggregate", to_node="aggregate_executor", reasoning=aggregation.reasoning, last = False)
        yield Status(f"Aggregated from {aggregation.collection_name}")

        # final return
        yield GenericAggregation(objects, metadata) 