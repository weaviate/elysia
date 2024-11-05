
from backend.util.logging import backend_print
from backend.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from backend.tree.objects import Returns, Text

class Summarizer:

    def __init__(self):
        self.summarizer = SummarizingExecutor()

    def summarize(self, user_prompt: str, available_information: Returns, **kwargs) -> str:
        summary = self.summarizer(user_prompt=user_prompt, available_information=available_information.to_llm_str())
        output = Text([summary], {})
        return output
    
    def __call__(self, user_prompt: str, available_information: Returns, **kwargs) -> str:
        return self.summarize(user_prompt, available_information, **kwargs)

class TextResponse:

    def __init__(self):
        self.text_response = TextResponseExecutor()

    def __call__(self, user_prompt: str, available_information: Returns, **kwargs) -> str:
        output = self.text_response(user_prompt=user_prompt, available_information=available_information.to_llm_str())
        return Text([output], {})