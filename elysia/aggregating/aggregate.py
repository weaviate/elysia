import dspy

from elysia.util.logging import backend_print
from elysia.util.parsing import update_current_message, format_aggregation_response

from elysia.tree import complex_lm
from elysia.tree.objects import Returns, Objects, Status, Warning, Error, Branch, TreeUpdate
from elysia.text.objects import Response, Code
from elysia.aggregating.objects import GenericAggregation
from elysia.aggregating.prompt_executors import AggregateExecutor

class AgenticAggregate:

    def __init__(self, 
                 aggregate_initialiser_filepath: str = "elysia/training/dspy_models/aggregate_initialiser/fewshot_k12.json", 
                 aggregate_executor_filepath: str = "elysia/training/dspy_models/aggregate_executor/fewshot_k12.json",
                 collection_names: list[str] = None):
        
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
            user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs
        ):
        
        data_queried = kwargs.get("data_queried", [])
        collection_information = kwargs.get("collection_information", [])
        current_message = kwargs.get("current_message", "")
        

        # Get some metadata about the collection
        self._find_previous_aggregations(available_information)

        # -- Step 2: Aggregate
        Branch({
            "name": "Aggregation Executor",
            "description": "Write code and aggregate the collection."
        })
        yield Status(f"Writing aggregation")

        # try:
        with dspy.context(lm = complex_lm):
            response, aggregation = self.aggregate_executor(
                user_prompt=user_prompt, 
                data_queried=data_queried, 
                collection_information=collection_information, 
                previous_reasoning=previous_reasoning,
                previous_aggregations=self.previous_aggregations
            )

        # except Exception as e:
        #     yield Error(f"Error in aggregating: {e}")

        # If the query is not possible, yield a generic retrieval and return nothing
        if aggregation is None:
            yield GenericAggregation([], {"collection_name": aggregation.collection_name, "impossible_prompts": [user_prompt]})
            return

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
        if message_update != "":
            yield Response([{"text": message_update}], {})
        yield Code([{"text": aggregation.code, "language": "python", "title": "Aggregation"}], {})
        yield TreeUpdate(from_node="aggregate", to_node="aggregate_executor", reasoning=aggregation.reasoning, last = False)
        yield Status(f"Aggregated from {aggregation.collection_name}")

        # final return
        yield GenericAggregation(objects, metadata) 