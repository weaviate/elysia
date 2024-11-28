import dspy
import json
from elysia.training.prompt_templates import construct_training_decision_prompt
from elysia.globals.reference import create_reference

class TrainingDecisionExecutor(dspy.Module):

    def __init__(self, available_tasks: list[dict] = None):
        self.router = dspy.Predict(construct_training_decision_prompt(available_tasks))
    
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
                idx: int = 0,
                **kwargs) -> tuple[dict, bool]:

        decision = self.router(
            task = kwargs.get("task", ""),
            user_prompt=user_prompt,
            reference=create_reference(),
            instruction=instruction,
            conversation_history=conversation_history,
            collection_information=collection_information,
            previous_reasoning=previous_reasoning,
            tree_count=tree_count,
            data_queried=data_queried,
            current_message=current_message,
            available_tasks=available_tasks,
            available_information=available_information,
            future_information=future_information,
            config = {"temperature": 0.7+0.01*idx}
        )

        return decision, False