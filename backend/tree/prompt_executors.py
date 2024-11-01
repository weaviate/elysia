import dspy
import json
from backend.tree.prompt_templates import DecisionPrompt

class DecisionExecutor(dspy.Module):

    def __init__(self):
        self.router = dspy.ChainOfThought(DecisionPrompt)
    
    def forward(self, 
                user_prompt: str, 
                instruction: str,
                available_tasks: list[dict], 
                completed_tasks: list[str]) -> tuple[dict, bool]:

        # convert available_tasks to a string
        available_tasks_str = json.dumps(available_tasks)

        decision = self.router(
            user_prompt=user_prompt,
            instruction=instruction,
            completed_tasks=completed_tasks,
            available_tasks=available_tasks_str
        )

        try:
            task = json.loads(decision.task)
        except Exception as e:
            print(f"Error reading routing output as JSON: {e}")
            return None, False

        try:
            completed = bool(eval(decision.user_will_be_satisfied))
        except Exception as e:
            print(f"Error reading completed output as boolean: {e}")
            return task, False

        return task["name"], completed

