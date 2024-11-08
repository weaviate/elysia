
from elysia.util.logging import backend_print
from elysia.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from elysia.tree.objects import Returns, Text

class Summarizer:

    def __init__(self):
        self.summarizer = SummarizingExecutor()

    def summarize(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs) -> str:
        summary = self.summarizer(
            user_prompt=user_prompt, 
            available_information=available_information.to_llm_str(),
            previous_reasoning=previous_reasoning,
            **kwargs
        )
        output = Text([summary], {})
        return output
    
    def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs) -> str:
        return self.summarize(user_prompt, available_information, previous_reasoning, **kwargs)

class TextResponse:

    def __init__(self):
        self.text_response = TextResponseExecutor()

    def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs) -> str:
        output = self.text_response(
            user_prompt=user_prompt, 
            available_information=available_information.to_llm_str(),
            previous_reasoning=previous_reasoning,
            **kwargs
        )
        return Text([output], {})
