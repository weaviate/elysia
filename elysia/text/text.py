
from elysia.util.logging import backend_print
from elysia.text.prompt_executors import SummarizingExecutor, TextResponseExecutor
from elysia.tree.objects import Returns, Text

class Summarizer:

    def __init__(self):
        self.summarizer = SummarizingExecutor()

    async def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs):

        conversation_history = kwargs.get("conversation_history", [])

        summary = self.summarizer(
            user_prompt=user_prompt, 
            available_information=available_information.to_llm_str(),
            previous_reasoning=previous_reasoning,
            conversation_history=conversation_history,
            **kwargs
        )
        output = Text([summary], {})
        yield output
    
class TextResponse:

    def __init__(self):
        self.text_response = TextResponseExecutor()

    async def __call__(self, user_prompt: str, available_information: Returns, previous_reasoning: dict = {}, **kwargs):

        conversation_history = kwargs.get("conversation_history", [])

        output = self.text_response(
            user_prompt=user_prompt, 
            available_information=available_information.to_llm_str(),
            previous_reasoning=previous_reasoning,
            conversation_history=conversation_history,
            **kwargs
        )

        yield Text([output], {})
