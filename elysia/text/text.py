
from elysia.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from elysia.tree.objects import Returns, Branch, Status
from elysia.text.objects import Response, Summary
from elysia.util.logging import backend_print

class Summarizer:

    def __init__(self):
        self.summarizer = SummarizingExecutor()

    async def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs):

        conversation_history = kwargs.get("conversation_history", [])

        yield Status("Summarising results")
        summary = self.summarizer(
            user_prompt=user_prompt, 
            available_information=available_information.to_llm_str(),
            previous_reasoning=previous_reasoning,
            conversation_history=conversation_history
        )
        
        output = Summary([{"text": summary.summary, "title": summary.subtitle}], {})
        yield output
    
class TextResponse:

    def __init__(self):
        self.text_response = TextResponseExecutor()

    async def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs):

        Branch({
            "name": "Text Response",
            "description": "Generate a chat response to the user's prompt.",
            "returns": "text"
        })

        current_message = kwargs.get("current_message", "")
        conversation_history = kwargs.get("conversation_history", [])

        yield Status("Crafting response")

        output = self.text_response(
            user_prompt=user_prompt, 
            available_information=available_information.to_llm_str(),
            previous_reasoning=previous_reasoning,
            conversation_history=conversation_history,
            current_message=current_message
        )

        yield Response([{"text": output}], {})