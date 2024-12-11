# Prompt Executors
from elysia.text.prompt_executors import SummarizingExecutor, TextResponseExecutor

# Objects
from elysia.tree.objects import Returns, TreeData, ActionData, DecisionData
from elysia.text.objects import Response, Summary
from elysia.api.objects import Status

# LLM
from elysia.dspy.cached_lm import LM

# 
import dspy

class Summarizer:

    def __init__(self, base_lm: LM, complex_lm: LM):
        self.summarizer = SummarizingExecutor()
        self.base_lm = base_lm
        self.complex_lm = complex_lm

    async def __call__(
        self, 
        tree_data: TreeData,
        action_data: ActionData,
        decision_data: DecisionData
    ):
        with dspy.context(lm=self.base_lm):
            summary = self.summarizer(
                user_prompt=tree_data.current_message, 
                available_information=decision_data.available_information.to_json(),
                previous_reasoning=tree_data.previous_reasoning,
                conversation_history=tree_data.conversation_history
            )
            
        yield Summary([{"text": summary.summary, "title": summary.subtitle}], {})
    
class TextResponse:

    def __init__(self, base_lm: LM, complex_lm: LM):
        self.text_response = TextResponseExecutor()
        self.base_lm = base_lm
        self.complex_lm = complex_lm

    async def __call__(
        self, 
        tree_data: TreeData,
        action_data: ActionData,
        decision_data: DecisionData
    ):

        yield Status("Crafting response")

        with dspy.context(self.base_lm):
            output = self.text_response(
                user_prompt=tree_data.user_prompt, 
                available_information=decision_data.available_information.to_json(),
                previous_reasoning=tree_data.previous_reasoning,
                conversation_history=tree_data.conversation_history,
                current_message=tree_data.current_message
            )

        output = "\n\n" + output

        yield Response([{"text": output}], {})
