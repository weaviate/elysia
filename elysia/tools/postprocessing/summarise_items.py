import dspy

from logging import Logger

from elysia.objects import Status, Tool
from elysia.tools.postprocessing.prompt_templates import ObjectSummaryPrompt
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager


class SummariseItems(Tool):
    def __init__(self, logger: Logger | None = None, **kwargs):
        super().__init__(
            name="query_postprocessing",
            description="""
            If the user has requested itemised summaries for retrieved objects,
            this tool summarises each object on an individual basis.
            """,
            status="",
            end=False,
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

        if "items_to_summarise" in tree_data.environment.hidden_environment:

            # get objects from hidden environment
            objects_list = tree_data.environment.hidden_environment[
                "items_to_summarise"
            ]

            for obj in objects_list:
                object_summariser = dspy.ChainOfThought(ObjectSummaryPrompt)
                yield Status(
                    f"Summarising {len(obj.objects)} objects from {obj.metadata['collection_name']}..."
                )

                summariser = await object_summariser.aforward(
                    objects=obj.to_json(), lm=base_lm
                )

                obj.add_summaries(summariser.summaries)

                yield obj

            # remove items from hidden environment
            tree_data.environment.hidden_environment.pop("items_to_summarise")
