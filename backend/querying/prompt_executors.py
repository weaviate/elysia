import dspy
from backend.querying.prompt_templates import QueryRewritingPrompt


class QueryRewriterExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_rewriting_prompt = dspy.ChainOfThought(QueryRewritingPrompt)

    def forward(self, user_prompt: str, previous_queries: list) -> str:
        return self.query_rewriting_prompt(user_prompt=user_prompt, previous_queries=previous_queries).query

