import dspy
import datetime
from typing import Any, Generator

from weaviate.classes.query import Filter, Sort
from weaviate.collections.classes.internal import QueryReturn
from weaviate.classes.aggregate import GroupByAggregate

from typing import Callable

from elysia.tree.objects import Returns
from elysia.globals.weaviate_client import client
from elysia.globals.reference import create_reference
from elysia.querying.prompt_templates import (
    construct_query_prompt, 
    ObjectSummaryPrompt,  
)
from elysia.util.logging import backend_print
from elysia.util.parsing import format_datetime

class QueryExecutor(dspy.Module):

    def __init__(self, collection_names: list[str] = None, return_types: list[str] = None):
        super().__init__()
        self.query_prompt = dspy.ChainOfThought(construct_query_prompt(collection_names, return_types))
        self.available_collections = collection_names
        self.available_return_types = return_types

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
        previous_queries: list, 
        data_queried: list,
        previous_reasoning: dict,
        collection_information: list,
        current_message: str
    ) -> Generator[Any, Any, Any]:

        # run query code generation
        try:
            prediction = self.query_prompt(
                user_prompt=user_prompt, 
                reference=create_reference(),
                previous_reasoning=previous_reasoning,
                collection_information=collection_information,
                previous_queries=previous_queries,
                current_message=current_message,
                data_queried=data_queried
            )
        except Exception as e:
            backend_print(f"Error in query creator prompt: {e}")
            # Return empty values when there's an error
            return QueryReturn(objects=[]), None

        try:
            is_query_possible = eval(prediction.is_query_possible)
            assert isinstance(is_query_possible, bool)
        except Exception as e:
            try:
                dspy.Assert(False, f"Error getting is_query_possible: {e}", target_module=self.query_prompt)
            except Exception as e:
                backend_print(f"Error getting is_query_possible: {e}")
                # Return empty values when there's an error
                return QueryReturn(objects=[]), None

        if not is_query_possible:
            return QueryReturn(objects=[]), None

        dspy.Suggest(
            prediction.code not in previous_queries,
            f"The query code you have produced: {prediction.code} has already been used. Please produce a new query code.",
            target_module=self.query_prompt
        )

        # catch any errors in query execution for dspy assert
        try:
            response = self._execute_code(prediction.code, prediction.collection_name)
        except Exception as e:

            try:
                # assert will raise an error if its failed multiple times
                dspy.Assert(False, f"Error executing query code:\n{prediction.code}\nERROR: {e}", target_module=self.query_prompt)
            except Exception as e:
                # in which case we just print the error and return 0 objects
                backend_print(f"Error executing query code: {e}")
                return QueryReturn(objects=[]), None

        return response, prediction

class ObjectSummaryExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.object_summary_prompt = dspy.ChainOfThought(ObjectSummaryPrompt)

    def forward(self, objects: list[dict]):
        prediction = self.object_summary_prompt(objects=objects)

        try:
            summary_list = eval(prediction.summaries)
        except Exception as e:
            dspy.Assert(False, f"Error converting summaries to list: {e}", target_module=self.object_summary_prompt)

        return summary_list, prediction