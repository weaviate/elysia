import dspy
from elysia.api.prompt_templates import TitleCreatorPrompt, ObjectRelevancePrompt

class TitleCreatorExecutor(dspy.Module):
    def __init__(self):
        self.title_creator_prompt = dspy.ChainOfThought(TitleCreatorPrompt)

    def __call__(self, text: str) -> str:
        return self.title_creator_prompt(text=text)

class ObjectRelevanceExecutor(dspy.Module):
    def __init__(self):
        self.object_relevance_prompt = dspy.ChainOfThought(ObjectRelevancePrompt)

    def __call__(self, user_prompt: str, objects: list[dict]) -> bool:
        return self.object_relevance_prompt(user_prompt=user_prompt, objects=objects)