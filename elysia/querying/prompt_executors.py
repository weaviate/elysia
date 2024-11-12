import dspy
import datetime

from weaviate.classes.query import Filter, Sort
from weaviate.collections.classes.internal import QueryReturn

from typing import Callable

from elysia.tree.objects import Returns
from elysia.globals.weaviate_client import client
from elysia.globals.reference import reference
from elysia.querying.prompt_templates import construct_query_initialiser_prompt, QueryCreatorPrompt
from elysia.util.logging import backend_print
from elysia.util.parsing import format_datetime

class QueryInitialiserExecutor(dspy.Module):

    def __init__(self, collection_names: list[str] = None, return_types: dict[str, str] = None):
        super().__init__()
        self.query_initialiser_prompt = dspy.ChainOfThought(construct_query_initialiser_prompt(collection_names, return_types))
        self.available_collections = collection_names
        self.available_return_types = return_types

    def forward(self, user_prompt: str, reference: str, previous_reasoning: dict, data_queried: list[str]) -> str:
        return self.query_initialiser_prompt(
            user_prompt=user_prompt,
            reference=reference,
            previous_reasoning=previous_reasoning,
            data_queried=data_queried,
            available_collections=self.available_collections,
            available_return_types=self.available_return_types
        )

class QueryExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_creator_prompt = dspy.ChainOfThought(QueryCreatorPrompt)

    def _create_query(
        self, 
        user_prompt: str, 
        collection_name: str,
        previous_reasoning: dict,
        available_information: Returns
    ):

        collection = client.collections.get(collection_name)

        # get example fields from collection
        example_field = collection.query.fetch_objects(limit=1).objects[0].properties
        for key in example_field:
            if isinstance(example_field[key], datetime.datetime):
                example_field[key] = example_field[key].isoformat()

        data_fields = list(example_field.keys())

        self._find_previous_queries(collection_name, available_information)

        # run creator to get query code
        query_output = self.query_creator(
            user_prompt=user_prompt,
            reference=reference,
            previous_queries=self.previous_queries,
            data_fields=data_fields,
            example_field=example_field,
            previous_reasoning=previous_reasoning
        )

        return query_output

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
        collection_name: str
    ) -> str:

        # run query code generation
        prediction = self.query_creator_prompt(
            user_prompt=user_prompt, 
            reference=reference,
            data_fields=data_fields, 
            example_field=example_field, 
            previous_queries=previous_queries,
            previous_reasoning=previous_reasoning
        )

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
                return QueryReturn(objects=[])


        return response, prediction.code
