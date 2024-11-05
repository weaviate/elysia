import dspy
import json
from backend.tree.prompt_templates import DecisionPrompt, InputPrompt
from backend.tree.objects import Returns

class DecisionExecutor(dspy.Module):

    def __init__(self):
        self.router = dspy.ChainOfThought(DecisionPrompt)
    
    def forward(self, 
                user_prompt: str, 
                instruction: str,
                available_tasks: list[dict], 
                available_information: Returns,
                conversation_history: list[dict],
                completed_tasks: list[str]) -> tuple[dict, bool]:

        # convert available_tasks to a string
        available_tasks_list = list(available_tasks.keys()) # provide the task names 
        available_tasks_str = str(available_tasks_list) + "\n" + json.dumps(available_tasks) # append the task descriptions

        decision = self.router(
            user_prompt=user_prompt,
            instruction=instruction,
            completed_tasks=completed_tasks,
            available_tasks=available_tasks_str,
            available_information=available_information.to_llm_str(),
            conversation_history=conversation_history
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

class InputExecutor(dspy.Module):

    def __init__(self):
        self.input_model = dspy.ChainOfThought(InputPrompt)

    def forward(self, task: str) -> list[str]:
        return self.input_model(task=task).subtasks
