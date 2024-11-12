import dspy
import datetime

from typing import Callable

from elysia.globals.weaviate_client import client
from elysia.querying.prompt_templates import construct_query_initialiser_prompt, QueryCreatorPrompt

class QueryInitialiserExecutor(dspy.Module):

    def __init__(self, collection_names: list[str] = None, return_types: list[str] = None):
        super().__init__()
        self.query_initialiser_prompt = dspy.ChainOfThought(construct_query_initialiser_prompt(collection_names, return_types))

    def forward(self, user_prompt: str, reference: str, previous_reasoning: dict, data_queried: list[str]) -> str:
        return self.query_initialiser_prompt(
            user_prompt=user_prompt,
            reference=reference,
            previous_reasoning=previous_reasoning,
            data_queried=data_queried
        )

class QueryCreatorExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_creator_prompt = dspy.ChainOfThought(QueryCreatorPrompt)

    def forward(self, user_prompt: str, reference: str, previous_queries: list, data_fields: list, example_field: dict, previous_reasoning: dict) -> str:

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

        return prediction
