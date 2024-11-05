import dspy
from backend.summarizing.prompt_templates import SummarizingPrompt

from backend.util.parsing import objects_dict_to_str
from backend.tree.objects import Returns
class SummarizingExecutor(dspy.Module):

    def __init__(self):
        super().__init__()
        self.summarizing_prompt = dspy.ChainOfThought(SummarizingPrompt)

    def forward(self, user_prompt: str, available_information: str) -> str:
        return self.summarizing_prompt(user_prompt=user_prompt, available_information=available_information).summary

