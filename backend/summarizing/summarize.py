
from backend.util.logging import backend_print
from backend.summarizing.prompt_executors import SummarizingExecutor

class Summarizer:

    def __init__(self):
        pass

    def summarize(self, user_prompt: str, completed_tasks: list[dict], **kwargs) -> str:
        summarizer = SummarizingExecutor()
        return summarizer(user_prompt=user_prompt, completed_tasks=completed_tasks), {}
    
    def __call__(self, user_prompt: str, completed_tasks: list[dict], **kwargs) -> str:
        return self.summarize(user_prompt, completed_tasks, **kwargs)
