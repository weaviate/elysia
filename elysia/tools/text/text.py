# Prompt Executors
#
import dspy
import dspy.predict

from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought

# LLM
from elysia.objects import Response, Tool
from elysia.tools.text.objects import TextWithTitle, TextWithCitations
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

        yield TextWithCitations(
            cited_texts=summary.cited_text,
            title=summary.subtitle,
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

        yield TextWithTitle(text=summary.summary, title=summary.subtitle)


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
            End the conversation. This should be used when the user has finished their query, or you have nothing more to do except reply.
            You should use this to answer conversational questions not related to other tools. But do not use this as a source of information.
            All information should be from the environment if answering a complex question or an explanation.
            If there is an error and you could not complete a task, use this tool to suggest a brief reason why.
            If, for example, there is a missing API key, then the user needs to add it to the settings (which you should inform them of).
            Or you cannot connect to weaviate, then the user needs to input their API keys in the settings.
            If there are no collections available, the user needs to analyze this in the 'data' tab.
            If there are other problems, and it looks like the user can fix it, then provide a suggestion.
            """,
            status="Writing response...",
            inputs={
                "text": {
                    "type": str,
                    "description": (
                        "The text to display to the user. Speak directly to them, do not say 'the user' or similar. "
                        "If you have achieved the goal, give a satisfying answer to their original prompt (user_prompt). "
                        "If you have not achieved the goal, explain. "
                        "If you know why, explain any shortcomings, maybe it was a limitation, a misunderstanding, a problem with the data. "
                        "Your suggestions should be useful to the user so they know if there is something possible to fix or do as a follow up. "
                        "Sometimes you can ask for clarification. "
                        "Be polite, professional, apologetic if necessary but above all - helpful! "
                        "Only provide a response in text, not using any variables or otherwise. "
                        "The text in this field is the DIRECT response to the user that will be displayed to them, and ending the conversation afterwards."
                    ),
                    "default": "",
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
