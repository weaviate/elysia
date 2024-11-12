import dspy
from elysia.api.prompt_templates import TitleCreatorPrompt

class TitleCreatorExecutor(dspy.Module):
    def __init__(self):
        self.title_creator_prompt = dspy.ChainOfThought(TitleCreatorPrompt)

    def __call__(self, text: str) -> str:
        return self.title_creator_prompt(text=text)
