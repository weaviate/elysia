# Prompt Executors
#
import dspy
import dspy.predict

from elysia.util.elysia_modules import ElysiaChainOfThought

# LLM
from elysia.objects import Response, Tool, Text
from elysia.tools.text.prompt_templates import (
    SummarizingPrompt,
    TextResponsePrompt,
    CitedSummarizingPrompt,
)

# Objects
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager


class CitedSummarizer(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="cited_summarize",
            description="""
            Summarize retrieved information for the user when all relevant data has been gathered.
            Provides a text response, and may end the conversation, but unlike text_response tool, can be used mid-conversation.
            Avoid for general questions where text_response is available.
            Summarisation text is directly displayed to the user.
            Most of the time, you can choose end_actions to be True to end the conversation with a summary.
            This is a good way to end the conversation.
            """,
            status="Summarizing...",
            inputs={},
            end=True,
        )

    async def is_tool_available(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        return not tree_data.environment.is_empty()

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        summarizer = ElysiaChainOfThought(
            CitedSummarizingPrompt,
            tree_data=tree_data,
            reasoning=False,
            impossible=False,
            environment=True,
            tasks_completed=True,
            message_update=False,
        )

        summary = await summarizer.aforward(
            lm=base_lm,
        )

        yield Text(
            "text_with_citations",
            objects=summary.cited_text,
            metadata={"title": summary.subtitle},
        )


class Summarizer(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="summarize",
            description="""
            Summarize retrieved information for the user when all relevant data has been gathered.
            Provides a text response, and may end the conversation, but unlike text_response tool, can be used mid-conversation.
            Avoid for general questions where text_response is available.
            Summarisation text is directly displayed to the user.
            This summarizer will also include citations in the summary, so relevant information must be in the environment for this tool.
            Most of the time, you can choose end_actions to be True to end the conversation with a summary.
            This is a good way to end the conversation.
            """,
            status="Summarizing...",
            inputs={},
            end=True,
        )

    async def is_tool_available(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        # when the environment is non empty
        return not tree_data.environment.is_empty()

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        summarizer = ElysiaChainOfThought(
            SummarizingPrompt,
            tree_data=tree_data,
            environment=True,
            tasks_completed=True,
            message_update=False,
        )

        summary = await summarizer.aforward(
            lm=base_lm,
        )

        yield Text(
            "text_with_title",
            objects=[{"text": summary.summary}],
            metadata={"title": summary.subtitle},
        )


class TextResponse(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="final_text_response",
            description="",
            status="Writing response...",
            inputs={},
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        text_response = ElysiaChainOfThought(
            TextResponsePrompt,
            tree_data=tree_data,
            environment=True,
            tasks_completed=True,
            message_update=False,
        )

        output = await text_response.aforward(
            lm=base_lm,
        )

        yield Response(text=output.response)


class FakeTextResponse(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="text_response",
            description="""
            End the conversation. Use when finished or to answer conversational questions. 
            Or use if you have relevant items in the environment to discuss.
            """,
            status="Writing response...",
            inputs={
                "text": {
                    "type": str,
                    "description": (
                        "The direct text response displayed to the user. "
                        "If successful, provide a satisfying answer to their original prompt. "
                        "If unsuccessful, explain why (limitations, misunderstandings, data issues) with actionable suggestions or ask for clarification. "
                        "If an error occurred, explain why and suggest fixes (e.g., add API keys in settings, analyze collections in 'data' tab). "
                        "Do not display or show any information unless it comes from the environment. "
                        "No repeating of any information in this step unless explicitly asked to do so."
                        "Plain text only. "
                    ),
                    "default": "",
                    "required": True,
                }
            },
            end=True,
        )

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager | None = None,
        **kwargs,
    ):
        yield Response(text=inputs["text"])
