import dspy

# dspy
from elysia.dspy.environment_of_thought import EnvironmentOfThought

# Prompt Templates
from elysia.api.services.prompt_templates import TitleCreatorPrompt, ObjectRelevancePrompt

class TitleCreatorExecutor(dspy.Module):
    def __init__(self):
        self.title_creator_prompt = EnvironmentOfThought(TitleCreatorPrompt)

    def __call__(self, text: str) -> str:
        return self.title_creator_prompt(text=text)

class ObjectRelevanceExecutor(dspy.Module):
    def __init__(self):
        self.object_relevance_prompt = EnvironmentOfThought(ObjectRelevancePrompt)

    def __call__(self, user_prompt: str, objects: list[dict]) -> bool:
        return self.object_relevance_prompt(user_prompt=user_prompt, objects=objects)