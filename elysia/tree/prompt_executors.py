import dspy
import json

# Prompt Templates
from elysia.tree.prompt_templates import (
    construct_decision_prompt, 
    InputPrompt
)

# Objects
from elysia.tree.objects import TreeData, DecisionData

# Globals
from elysia.globals.reference import create_reference

class DecisionExecutor(dspy.Module):

    def __init__(self, available_tasks: list[dict] = None):
        self.router = dspy.ChainOfThought(construct_decision_prompt(available_tasks))
    
    def forward(self, 
                tree_data: TreeData,
                decision_data: DecisionData, 
                idx: int = 0) -> tuple[dict, bool]:

        decision = self.router(
            user_prompt=tree_data.user_prompt,
            instruction=decision_data.instruction,
            reference=create_reference(),
            conversation_history=tree_data.conversation_history,
            previous_reasoning=tree_data.previous_reasoning,
            tree_count=decision_data.tree_count_string(),
            data_queried=tree_data.data_queried_string(),
            current_message=tree_data.current_message,
            available_tasks=decision_data.available_tasks,
            available_information=decision_data.available_information.to_llm_str(),
            future_information=decision_data.future_information,
            config={"temperature": 0.7+0.01*idx} # ensures randomness in LLM
        )

        # assert that the task name is correct
        dspy.Assert(decision.task in decision_data.available_tasks, 
                    f"""Decision task is not in available tasks: 
                    {decision.task} not in {decision_data.available_tasks}
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
