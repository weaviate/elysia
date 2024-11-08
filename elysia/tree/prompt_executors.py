import dspy
import json
from elysia.tree.prompt_templates import construct_decision_prompt, InputPrompt
from elysia.tree.objects import Returns
from elysia.globals.reference import reference as default_reference

class DecisionExecutor(dspy.Module):

    def __init__(self, available_tasks: list[dict] = None):
        self.router = dspy.ChainOfThought(construct_decision_prompt(available_tasks))
    
    def forward(self, 
                user_prompt: str, 
                instruction: str,
                available_tasks: list[dict], 
                available_information: str,
                conversation_history: list[dict],
                completed_tasks: list[str],
                data_queried: list[str],
                decision_tree: dict,
                previous_reasoning: dict,
                reference: dict = default_reference,
                idx: int = 0) -> tuple[dict, bool]:

        # convert available_tasks to a string
        available_tasks_list = list(available_tasks.keys()) # provide the task names 
        # available_tasks_str = str(available_tasks_list) + "\n" + json.dumps(available_tasks) # append the task descriptions
        available_tasks_str = json.dumps(available_tasks)

        decision_tree_str = json.dumps(decision_tree)

        data_queried_str = ""
        for collection_name, num_items in data_queried.items():
            data_queried_str += f" - {collection_name}: {num_items} objects retrieved {'(empty - either no objects for this prompt or incorrect query)' if num_items == 0 else ''}\n"
        
        decision = self.router(
            user_prompt=user_prompt,
            reference=reference,
            instruction=instruction,
            completed_tasks=completed_tasks,
            available_tasks=available_tasks_str,
            decision_tree=decision_tree_str,
            available_information=available_information,
            data_queried=data_queried_str,
            conversation_history=conversation_history,
            previous_reasoning=previous_reasoning,
            config = {"temperature": 0.7+0.01*idx}
        )

        # assert that the task name is correct
        dspy.Assert(decision.task in available_tasks_list, 
                    f"""Decision task is not in available tasks: 
                    {decision.task} not in {available_tasks_list}
                    Ensure that the task name is correct and that the task exists in the available_tasks field.""")


        try:
            completed = bool(eval(decision.all_actions_completed))
        except Exception as e:
            print(f"Error reading completed output as boolean: {e}")
            return decision, False

        return decision, completed

class InputExecutor(dspy.Module):

    def __init__(self):
        self.input_model = dspy.ChainOfThought(InputPrompt)

    def forward(self, task: str) -> list[str]:
        return self.input_model(task=task).subtasks
