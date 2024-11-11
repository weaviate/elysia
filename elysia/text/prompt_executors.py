import dspy
from elysia.text.prompt_templates import SummarizingPrompt, TextResponsePrompt
from elysia.globals.reference import reference

class SummarizingExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.summarizing_prompt = dspy.ChainOfThought(SummarizingPrompt)

    def forward(self, user_prompt: str, available_information: str, previous_reasoning: dict) -> str:
        return self.summarizing_prompt(
            user_prompt=user_prompt, 
            available_information=available_information,
            previous_reasoning=previous_reasoning,
            reference=reference
        ).summary

class TextResponseExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.text_response_prompt = dspy.ChainOfThought(TextResponsePrompt)

    def forward(self, user_prompt: str, available_information: str, previous_reasoning: dict) -> str:
        return self.text_response_prompt(
            user_prompt=user_prompt, 
            available_information=available_information,
            previous_reasoning=previous_reasoning,
            reference=reference
        ).response