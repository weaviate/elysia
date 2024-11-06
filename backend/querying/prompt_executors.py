import dspy
from backend.querying.prompt_templates import QueryCreatorPrompt

class QueryCreatorExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_creator_prompt = dspy.ChainOfThought(QueryCreatorPrompt)

    def forward(self, user_prompt: str, reference: str, data_fields: list, example_field: dict, previous_queries: list) -> str:
        return self.query_creator_prompt(
            user_prompt=user_prompt, 
            reference=reference,
            data_fields=data_fields, 
            example_field=example_field, 
            previous_queries=previous_queries
        ).code

