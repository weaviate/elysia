import datetime

import dspy
from weaviate.classes.query import Filter, Sort
from weaviate.collections.classes.internal import QueryReturn

from elysia.globals.weaviate_client import client
from elysia.globals.reference import reference
from elysia.util.logging import backend_print
from elysia.querying.prompt_executors import QueryCreatorExecutor, QueryInitialiserExecutor
from elysia.tree.objects import Returns, Objects, Status
from elysia.querying.objects import GenericRetrieval, MessageRetrieval, ConversationRetrieval, TicketRetrieval
from elysia.util.parsing import format_datetime

class AgenticQuery:

    def __init__(self, filepath: str = "elysia/training/dspy_models/agentic_query/fewshot_k12.json", collection_names: list[str] = None, return_types: list[str] = None): 

        self.query_initialiser = QueryInitialiserExecutor(collection_names, return_types).activate_assertions()        
        self.query_creator = QueryCreatorExecutor().activate_assertions()

        if len(filepath) > 0:
            self.query_creator.load(filepath)

    def _find_previous_queries(self, collection_name: str, available_information: Returns):

        self.previous_queries = []
        if collection_name in available_information.retrieved:
            metadata = available_information.retrieved[collection_name].metadata
            if "previous_queries" in metadata:
                self.previous_queries.extend(metadata["previous_queries"])

    def _initialise_query(self, user_prompt: str, previous_reasoning: dict, data_queried: list[str]):

        # run initialiser to get collection name and return type
        initialiser = self.query_initialiser(
            user_prompt=user_prompt,
            reference=reference,
            previous_reasoning=previous_reasoning,
            data_queried=data_queried
        )

        return initialiser.collection_name, initialiser.return_type, initialiser.reasoning

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
        try:
            return eval(query_code)
        except Exception as e:
            backend_print(f"Error executing query code: {e}, returning 0 objects")
            return QueryReturn(objects=[])

    async def query(self, user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs):

        data_queried = kwargs.get("data_queried", [])

        collection_name, return_type, reasoning = self._initialise_query(user_prompt, previous_reasoning, data_queried)
        yield Status(f"Writing query for collection: {collection_name}")

        # add reasoning from initialiser to previous reasoning
        previous_reasoning["query_initialiser"] = reasoning

        query_output = self._create_query(user_prompt, collection_name, previous_reasoning, available_information)
        yield Status(f"Querying {collection_name}")

        response = self._execute_code(query_output.code, collection_name)
        yield Status(f"Retrieved {len(response.objects)} objects from {collection_name}")

        if return_type == "conversation":
            yield ConversationRetrieval(response, {"previous_queries": self.previous_queries, "collection_name": collection_name})
        elif return_type == "message":
            yield MessageRetrieval(response, {"previous_queries": self.previous_queries, "collection_name": collection_name})
        elif return_type == "ticket":
            yield TicketRetrieval(response, {"previous_queries": self.previous_queries, "collection_name": collection_name})
        else:
            yield GenericRetrieval(response, {"previous_queries": self.previous_queries, "collection_name": collection_name})

    def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs):
        return self.query(user_prompt, available_information, previous_reasoning, **kwargs)
