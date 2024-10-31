import dspy
from backend.actions.prompt_templates import QueryRewritingPrompt


class QueryRewriterExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.query_rewriting_prompt = QueryRewritingPrompt

    def forward(self, user_prompt: str) -> str:
        return self.query_rewriting_prompt(user_prompt=user_prompt).query

