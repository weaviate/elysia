import dspy
from elysia.querying.prompt_templates import QueryCreatorPrompt

class QueryCreatorExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_creator_prompt = dspy.ChainOfThought(QueryCreatorPrompt)

    def forward(self, user_prompt: str, reference: str, data_fields: list, example_field: dict, previous_queries: list) -> str:
        prediction = self.query_creator_prompt(
            user_prompt=user_prompt, 
            reference=reference,
            data_fields=data_fields, 
            example_field=example_field, 
            previous_queries=previous_queries
        )

        dspy.Suggest(
            prediction.code not in previous_queries,
            f"The query code you have produced: {prediction.code} has already been used. Please produce a new query code.",
            target_module=self.query_creator_prompt
        )

        return prediction.code

