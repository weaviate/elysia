import dspy
import datetime
from typing import Any, Generator

from weaviate.classes.query import Filter, Sort
from weaviate.collections.classes.internal import QueryReturn
from weaviate.collections.classes.aggregate import AggregateGroupByReturn, AggregateReturn

from typing import Callable

from elysia.tree.objects import Returns
from elysia.globals.weaviate_client import client
from elysia.globals.reference import reference
from elysia.querying.prompt_templates import (
    construct_query_initialiser_prompt, 
    QueryCreatorPrompt, 
    ObjectSummaryPrompt, 
    AggregateCollectionPrompt, 
    PropertyGroupingPrompt
)
from elysia.util.logging import backend_print
from elysia.util.parsing import format_datetime

class QueryInitialiserExecutor(dspy.Module):

    def __init__(self, collection_names: list[str] = None, return_types: dict[str, str] = None):
        super().__init__()
        self.query_initialiser_prompt = dspy.ChainOfThought(construct_query_initialiser_prompt(collection_names, return_types))
        self.available_collections = collection_names
        self.available_return_types = return_types

    def forward(self, user_prompt: str, reference: str, previous_reasoning: dict, data_queried: list[str], current_message: str) -> str:
        return self.query_initialiser_prompt(
            user_prompt=user_prompt,
            reference=reference,
            previous_reasoning=previous_reasoning,
            data_queried=data_queried,
            available_collections=self.available_collections,
            available_return_types=self.available_return_types,
            current_message=current_message
        )
    
class PropertyGroupingExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.property_grouping_prompt = dspy.ChainOfThought(PropertyGroupingPrompt)

    def forward(
        self, 
        user_prompt: str, 
        reference: str, 
        previous_reasoning: dict, 
        data_fields: list[str], 
        example_field: dict,
        current_message: str
    ) -> str:
        return self.property_grouping_prompt(
            user_prompt=user_prompt,
            reference=reference,
            previous_reasoning=previous_reasoning,
            data_fields=data_fields,
            example_field=example_field,
            current_message=current_message
        )

class QueryExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_creator_prompt = dspy.ChainOfThought(QueryCreatorPrompt)

    def _execute_code(self, query_code: str, collection_name: str) -> dict:

        collection = client.collections.get(collection_name)

        if query_code.startswith("```python") and query_code.endswith("```"):
            query_code = query_code[8:-3]
        elif query_code.startswith("```") and query_code.endswith("```"):
            query_code = query_code[3:-3]
   
        return eval(query_code)

    def forward(
        self, 
        user_prompt: str, 
        reference: str, 
        previous_queries: list, 
        data_fields: list, 
        example_field: dict, 
        previous_reasoning: dict,
        collection_name: str,
        collection_metadata: dict,
        current_message: str
    ) -> Generator[Any, Any, Any]:

        # run query code generation
        try:
            prediction = self.query_creator_prompt(
                user_prompt=user_prompt, 
                reference=reference,
                data_fields=data_fields, 
                example_field=example_field, 
                previous_queries=previous_queries,
                previous_reasoning=previous_reasoning,
                collection_metadata=collection_metadata,
                current_message=current_message
            )
        except Exception as e:
            backend_print(f"Error in query creator prompt: {e}")
            # Return empty values when there's an error
            return QueryReturn(objects=[]), None, None, False

        try:
            is_query_possible = bool(prediction.is_query_possible)
        except Exception as e:
            try:
                dspy.Assert(False, f"Error getting is_query_possible: {e}", target_module=self.query_creator_prompt)
            except Exception as e:
                backend_print(f"Error getting is_query_possible: {e}")
                # Return empty values when there's an error
                return QueryReturn(objects=[]), None, None, False

        if not is_query_possible:
            return QueryReturn(objects=[]), None, None, False

        dspy.Suggest(
            prediction.code not in previous_queries,
            f"The query code you have produced: {prediction.code} has already been used. Please produce a new query code.",
            target_module=self.query_creator_prompt
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, collection_name)
        except Exception as e:

            try:
                # assert will raise an error if its failed multiple times
                dspy.Assert(False, f"Error executing query code:\n{prediction.code}\nERROR: {e}", target_module=self.query_creator_prompt)
            except Exception as e:
                # in which case we just print the error and return 0 objects
                backend_print(f"Error executing query code: {e}")
                return QueryReturn(objects=[]), None, None, False

        return response, prediction

class AggregateCollectionExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.aggregate_collection_prompt = dspy.ChainOfThought(AggregateCollectionPrompt)

    def _execute_code(self, aggregation_code: str, collection_name: str) -> dict:

        collection = client.collections.get(collection_name)

        if aggregation_code.startswith("```python") and aggregation_code.endswith("```"):
            aggregation_code = aggregation_code[8:-3]
        elif aggregation_code.startswith("```") and aggregation_code.endswith("```"):
            aggregation_code = aggregation_code[3:-3]
   
        return eval(aggregation_code)

    def forward(
        self, 
        user_prompt: str, 
        reference: str, 
        data_fields: list, 
        example_field: dict, 
        previous_reasoning: dict, 
        collection_name: str
    ) -> str:
        
        prediction = self.aggregate_collection_prompt(
            user_prompt=user_prompt, 
            reference=reference, 
            previous_reasoning=previous_reasoning,
            data_fields=data_fields, 
            example_field=example_field, 
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, collection_name)
        except Exception as e:

            try:
                # assert will raise an error if its failed multiple times
                dspy.Assert(False, f"Error executing aggregation code:\n{prediction.code}\nERROR: {e}", target_module=self.aggregate_collection_prompt)
            except Exception as e:
                # in which case we just print the error and return 0 objects
                backend_print(f"Error executing aggregation code: {e}")
                return AggregateReturn(properties={}, total_count=0), None, None
            
        return response, prediction.code, prediction.text_return

class ObjectSummaryExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.object_summary_prompt = dspy.ChainOfThought(ObjectSummaryPrompt)

    def forward(self, objects: list[dict]) -> list[str]:
        summaries = self.object_summary_prompt(objects=objects).summaries

        try:
            summary_list = eval(summaries)
        except Exception as e:
            dspy.Assert(False, f"Error converting summaries to list: {e}", target_module=self.object_summary_prompt)

        return summary_list