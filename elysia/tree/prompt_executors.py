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
                future_tasks: dict,
                previous_reasoning: dict,
                collection_names: list[str],
                reference: dict = default_reference,
                current_message: str = "",
                tree_count: str = "",
                idx: int = 0) -> tuple[dict, bool]:

        # convert available_tasks to a string
        # available_tasks_list = list(available_tasks.keys()) # provide the task names 
        # available_tasks_str = str(available_tasks_list) + "\n" + json.dumps(available_tasks) # append the task descriptions
        available_tasks_str = json.dumps(available_tasks)
        decision_tree_str = json.dumps(decision_tree)

        decision = self.router(
            user_prompt=user_prompt,
            reference=reference,
            instruction=instruction,
            completed_tasks=completed_tasks,
            available_tasks=available_tasks_str,
            decision_tree=decision_tree_str,
            available_information=available_information,
            collection_names=collection_names,
            data_queried=data_queried,
            future_information=future_tasks,
            conversation_history=conversation_history,
            previous_reasoning=previous_reasoning,
            current_message=current_message,
            tree_count=tree_count,
            config = {"temperature": 0.7+0.01*idx}
        )

        # assert that the task name is correct
        dspy.Assert(decision.task in available_tasks, 
                    f"""Decision task is not in available tasks: 
                    {decision.task} not in {available_tasks}
                    Ensure that the task name is correct and that the task exists in the available_tasks field.""")


        try:
            completed = eval(decision.all_actions_completed)
            assert isinstance(completed, bool)
        except Exception as e:
            print(f"Error reading completed output as boolean: {e}")
            return decision, False

        return decision, completed

class InputExecutor(dspy.Module):

    def __init__(self):
        self.input_model = dspy.ChainOfThought(InputPrompt)

    def forward(self, task: str) -> list[str]:
        return self.input_model(task=task).subtasks
