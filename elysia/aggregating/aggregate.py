import datetime
import dspy

from weaviate.classes.aggregate import GroupByAggregate

from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference
from elysia.util.logging import backend_print
from elysia.util.parsing import update_current_message, format_aggregation_response
from elysia.util.collection_metadata import get_collection_data_types

from elysia.querying.prompt_executors import (
    QueryExecutor, 
    QueryInitialiserExecutor, 
    PropertyGroupingExecutor, 
    ObjectSummaryExecutor
)
from elysia.tree.objects import Returns, Objects, Status, Warning, Error, Branch, TreeUpdate
from elysia.text.objects import Response, Code
from elysia.querying.objects import GenericRetrieval
from elysia.aggregating.objects import GenericAggregation
from elysia.aggregating.prompt_executors import AggregateInitialiserExecutor, AggregateExecutor

class AgenticAggregate:

    def __init__(self, 
                 aggregate_initialiser_filepath: str = "elysia/training/dspy_models/aggregate_initialiser/fewshot_k12.json", 
                 aggregate_executor_filepath: str = "elysia/training/dspy_models/aggregate_executor/fewshot_k12.json",
                 collection_names: list[str] = None):
        
        self.collection_names = collection_names
        self.aggregate_initialiser = AggregateInitialiserExecutor(self.collection_names).activate_assertions()
        self.aggregate_executor = AggregateExecutor().activate_assertions()
    
    def _find_previous_aggregations(self, collection_name: str, available_information: Returns):

        self.previous_aggregations = []
        if collection_name in available_information.retrieved:
            metadata = available_information.retrieved[collection_name].metadata
            if "previous_aggregations" in metadata:
                self.previous_aggregations.extend(metadata["previous_aggregations"])       
    
    def _get_collection_fields(self, collection_name: str):
        example_field = client.collections.get(collection_name).query.fetch_objects(limit=1).objects[0].properties
        for key in example_field:
            if isinstance(example_field[key], datetime.datetime):
                example_field[key] = example_field[key].isoformat()
            elif not isinstance(example_field[key], str):
                example_field[key] = str(example_field[key])

        data_types = get_collection_data_types(collection_name)

        return data_types, example_field

    async def __call__(self, 
            user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs
        ):
        
        data_queried = kwargs.get("data_queried", [])
        current_message = kwargs.get("current_message", "")
        
        # -- Step 1: Determine collection
        Branch({
            "name": "Aggregation Initialiser",
            "description": "Determine the collection to aggregate over."
        })
        yield Status(f"Initialising aggregation")


        # try:

        initialiser = self.aggregate_initialiser(
            user_prompt=user_prompt, 
            data_queried=data_queried, 
            current_message=current_message, 
            previous_reasoning=previous_reasoning
        )

        current_message, message_update = update_current_message(current_message, initialiser.text_return)

        # Yield results to front end
        yield TreeUpdate(from_node="aggregate", to_node="aggregate_initialiser", reasoning=initialiser.reasoning, last = False)
        if message_update != "":
            yield Response([{"text": message_update}], {})

        # except Exception as e:
        #     yield Error(f"Error in initialising aggregation: {e}")


        # Get some metadata about the collection
        self._find_previous_aggregations(initialiser.collection_name, available_information)
        data_types, example_field = self._get_collection_fields(initialiser.collection_name)

        # -- Step 2: Aggregate
        Branch({
            "name": "Aggregation Executor",
            "description": "Write code and aggregate the collection."
        })
        yield Status(f"Aggregating {initialiser.collection_name}")

        # try:
        response, aggregation = self.aggregate_executor(
            user_prompt=user_prompt, 
            data_types=data_types, 
            example_field=example_field, 
            previous_reasoning=previous_reasoning,
            previous_aggregations=self.previous_aggregations, 
            collection_name=initialiser.collection_name
        )

        # except Exception as e:
        #     yield Error(f"Error in aggregating: {e}")

        # If the query is not possible, yield a generic retrieval and return nothing
        if aggregation is None:
            yield GenericAggregation([], {"collection_name": initialiser.collection_name, "impossible_prompts": [user_prompt]})
            return

        current_message, message_update = update_current_message(current_message, aggregation.text_return)

        # return values
        objects = [
            {initialiser.collection_name: format_aggregation_response(response)}
        ]
        metadata = {
            "collection_name": initialiser.collection_name,
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
        yield TreeUpdate(from_node="aggregate_initialiser", to_node="aggregate_executor", reasoning=aggregation.reasoning, last = False)
        yield Status(f"Aggregated from {initialiser.collection_name}")

        # final return
        yield GenericAggregation(objects, metadata) 