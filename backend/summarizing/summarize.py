
from backend.util.logging import backend_print
from backend.summarizing.prompt_executors import SummarizingExecutor
from backend.tree.objects import Returns, Text

class Summarizer:

    def __init__(self):
        pass

    def summarize(self, user_prompt: str, available_information: Returns, **kwargs) -> str:
        summarizer = SummarizingExecutor()
        summary = summarizer(user_prompt=user_prompt, available_information=available_information.to_llm_str())
        output = Text([summary], {})
        return output
    
    def __call__(self, user_prompt: str, available_information: Returns, **kwargs) -> str:
        return self.summarize(user_prompt, available_information, **kwargs)
