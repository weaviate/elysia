import dspy
import json

# Prompt Templates
from elysia.tree.prompt_templates import (
    construct_decision_prompt, 
    InputPrompt
)

# Objects
from elysia.tree.objects import TreeData, DecisionData, ActionData

# Globals
from elysia.globals.reference import create_reference

class DecisionExecutor(dspy.Module):

    def __init__(self, available_tasks: list[dict] = None):
        self.router = dspy.ChainOfThought(construct_decision_prompt(available_tasks))
    
    def forward(self, 
                user_prompt: str,
                instruction: str,
                conversation_history: list[dict],
                collection_information: list,
                previous_reasoning: dict,
                tree_count: str,
                data_queried: list,
                current_message: str,
                available_tasks: list[dict],
                available_information: list,
                future_information: list,
                idx: int = 0) -> tuple[dict, bool]:

        decision = self.router(
            user_prompt=user_prompt,
            instruction=instruction,
            reference=create_reference(),
            conversation_history=conversation_history,
            collection_information=collection_information,
            previous_reasoning=previous_reasoning,
            tree_count=tree_count,
            data_queried=data_queried,
            current_message=current_message,
            available_tasks=available_tasks,
            available_information=available_information,
            future_information=future_information,
            config={"temperature": 0.7+0.01*idx} # ensures randomness in LLM
        )
        completed = decision.all_actions_completed
        return decision, completed

class InputExecutor(dspy.Module):

    def __init__(self):
        self.input_model = dspy.ChainOfThought(InputPrompt)

    def forward(self, task: str) -> list[str]:
        return self.input_model(task=task).subtasks
