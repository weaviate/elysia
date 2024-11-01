import dspy
from backend.summarizing.prompt_templates import SummarizingPrompt


class SummarizingExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.summarizing_prompt = dspy.ChainOfThought(SummarizingPrompt)

    def _collect_objects(self, completed_tasks: list[dict]) -> list[dict]:
        retrieved_objects = []
        for task in completed_tasks:
            if task["id"] == "collection":
                retrieved_objects.append(task["result"])
        return retrieved_objects

    def forward(self, user_prompt: str, completed_tasks: list[dict]) -> str:
        retrieved_objects = self._collect_objects(completed_tasks)
        
        return self.summarizing_prompt(user_prompt=user_prompt, retrieved_objects=retrieved_objects).summary

