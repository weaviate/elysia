import dspy
from backend.routing.prompt_templates import RoutingPrompt

class RoutingExecutor(dspy.Module):

    def __init__(self):
        self.router = dspy.ChainOfThought(RoutingPrompt)
    
    def forward(self, user_prompt: str, description: str, options: list[str]) -> str:
        return self.router(
            user_prompt=user_prompt,
            property_description=description,
            property_options=options
        ).property


