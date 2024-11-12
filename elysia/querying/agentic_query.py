import datetime

import dspy

from elysia.globals.weaviate_client import client
from elysia.globals.reference import reference
from elysia.util.logging import backend_print
from elysia.querying.prompt_executors import QueryExecutor, QueryInitialiserExecutor
from elysia.tree.objects import Returns, Objects, Status
from elysia.querying.objects import GenericRetrieval, MessageRetrieval, ConversationRetrieval, TicketRetrieval

class AgenticQuery:

    def __init__(self, filepath: str = "elysia/training/dspy_models/agentic_query/fewshot_k12.json", collection_names: list[str] = None, return_types: list[str] = None): 

        self.query_initialiser = QueryInitialiserExecutor(collection_names, return_types).activate_assertions()        
        self.querier = QueryExecutor().activate_assertions()

        if len(filepath) > 0:
            self.querier.load(filepath)

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

    
    async def query(self, user_prompt: str, available_information: Returns, previous_reasoning: dict, **kwargs):

        data_queried = kwargs.get("data_queried", [])

        yield Status(f"Determining which collection to query")
        collection_name, return_type, reasoning = self._initialise_query(user_prompt, previous_reasoning, data_queried)
        
        # get example fields from collection
        example_field = client.collections.get(collection_name).query.fetch_objects(limit=1).objects[0].properties
        for key in example_field:
            if isinstance(example_field[key], datetime.datetime):
                example_field[key] = example_field[key].isoformat()

        data_fields = list(example_field.keys())

        self._find_previous_queries(collection_name, available_information)

        # add reasoning from initialiser to previous reasoning just for query step
        previous_reasoning["query_initialiser"] = reasoning

        yield Status(f"Querying {collection_name}")
        response, code = self.querier(
            user_prompt, 
            reference, 
            self.previous_queries, 
            data_fields, 
            example_field, 
            previous_reasoning,
            collection_name
        )
        yield Status(f"Retrieved {len(response.objects)} objects from {collection_name}")

        self.previous_queries.append(code)

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
