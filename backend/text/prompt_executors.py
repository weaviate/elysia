import dspy
from backend.text.prompt_templates import SummarizingPrompt, TextResponsePrompt
from backend.globals.reference import reference

class SummarizingExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.summarizing_prompt = dspy.ChainOfThought(SummarizingPrompt)

    def forward(self, user_prompt: str, available_information: str) -> str:
        return self.summarizing_prompt(
            user_prompt=user_prompt, 
            available_information=available_information,
            reference=reference
        ).summary

class TextResponseExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.text_response_prompt = dspy.ChainOfThought(TextResponsePrompt)

    def forward(self, user_prompt: str, available_information: str) -> str:
        return self.text_response_prompt(
            user_prompt=user_prompt, 
            available_information=available_information,
            reference=reference
        ).response
