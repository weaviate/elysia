from typing import Union
from logging import Logger

from rich import print
from rich.panel import Panel

import dspy

from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought
from elysia.objects import Response, Status, Tool, Error
from elysia.tools.retrieval.objects import Aggregation
from elysia.tools.retrieval.prompt_templates import (
    AggregationPrompt,
    construct_aggregation_output_prompt,
)
from elysia.tools.retrieval.util import (
    execute_weaviate_aggregation,
    QueryError,
    AggregationOutput,
    VectorisedAggregationOutput,
)
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.util.objects import TrainingUpdate, FewShotExamples
from elysia.util.parsing import format_aggregation_response


class Aggregate(Tool):
    def __init__(
        self,
        logger: Logger | None = None,
        **kwargs,
    ):
        self.logger = logger

        super().__init__(
            name="aggregate",
            description="""
            Query the knowledge base specifically for aggregation queries.
            Performs calculations (counting, averaging, summing, etc.) and provides summary statistics on data.
            It can group data by properties and apply filters directly, without needing a prior query.
            Aggregation queries can be filtered.
            This can be applied directly on any collections in the schema.
            Use this tool when you need counts, sums, averages, or other summary statistics on properties in the collections.
            'aggregate' should be considered the first choice for tasks involving counting, summing, averaging, 
            or other statistical operations, even when filtering is required.
            """,
            status="Aggregating...",
            inputs={
                "collection_names": {
                    "description": "the names of the collections that are most relevant to the aggregation. If you are unsure, give all collections.",
                    "type": list[str],
                    "default": [],
                },
            },
            end=False,
        )

    async def is_tool_available(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
    ) -> bool:
        """
        Only available when there is a Weaviate connection.
        If this tool is not available, inform the user that they need to set the WCD_URL and WCD_API_KEY in the settings.
        """
        return client_manager.is_client

    def _find_previous_aggregations(
        self, environment: dict, collection_names: list[str]
    ):
        previous_aggregations = {}
        for collection_name in collection_names:
            if (
                "aggregate" in environment
                and collection_name in environment["aggregate"]
            ):
                for aggregation in environment["aggregate"][collection_name]:
                    if "code" in aggregation and "text" in aggregation["code"]:
                        if collection_name in previous_aggregations:
                            previous_aggregations[collection_name].append(
                                aggregation["code"]["text"]
                            )
                        else:
                            previous_aggregations[collection_name] = [
                                aggregation["code"]["text"]
                            ]

        return previous_aggregations

    def _fix_collection_names(self, collection_names: list[str], schemas: dict):
        return [
            collection_name
            for collection_name in list(schemas.keys())
            if collection_name.lower()
            in [
                schema_collection_name.lower()
                for schema_collection_name in collection_names
            ]
        ]

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        client_manager: ClientManager,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        **kwargs,
    ):

        if self.logger:
            self.logger.debug("Aggregation Tool called!")
            self.logger.debug("Arguments received:")
            self.logger.debug(f"{inputs}")

        # Find matching collection names (to match case sensitivity)
        schemas = tree_data.output_collection_metadata(with_mappings=True)
        if inputs["collection_names"] == [] or inputs["collection_names"] is None:
            collection_names = list(schemas.keys())
        else:
            collection_names = self._fix_collection_names(
                inputs["collection_names"], schemas
            )

        # Set up the dspy model
        aggregation_prompt = AggregationPrompt

        # check if collections are vectorised
        vectorised_by_collection = {
            collection_name: (
                schemas[collection_name]["vectorizer"] is not None
                or (
                    schemas[collection_name]["named_vectors"] is not None
                    and schemas[collection_name]["named_vectors"] != []
                )
            )
            for collection_name in collection_names
        }

        if any(vectorised_by_collection.values()):
            aggregation_prompt = aggregation_prompt.append(
                name="aggregation_queries",
                type_=Union[list[VectorisedAggregationOutput], None],
                field=dspy.OutputField(
                    desc=construct_aggregation_output_prompt(True),
                ),
            )
        else:
            aggregation_prompt = aggregation_prompt.append(
                name="aggregation_queries",
                type_=Union[list[AggregationOutput], None],
                field=dspy.OutputField(
                    desc=construct_aggregation_output_prompt(False),
                ),
            )

        aggregate_generator = ElysiaChainOfThought(
            aggregation_prompt,
            tree_data=tree_data,
            environment=True,
            collection_schemas=True,
            tasks_completed=True,
            message_update=True,
            collection_names=inputs["collection_names"],
            reasoning=tree_data.settings.COMPLEX_USE_REASONING,
        )

        if self.logger:
            self.logger.info(
                f"Any collections vectorised: {any(vectorised_by_collection.values())}"
            )
            self.logger.info(f"Collection names received: {collection_names}")

        # Get some metadata about the collection
        previous_aggregations = self._find_previous_aggregations(
            tree_data.environment.environment, collection_names
        )

        # Generate query with LLM
        try:
            if tree_data.settings.USE_FEEDBACK:
                aggregation, example_uuids = (
                    await aggregate_generator.aforward_with_feedback_examples(
                        feedback_model="aggregate",
                        client_manager=client_manager,
                        base_lm=base_lm,
                        complex_lm=complex_lm,
                        available_collections=collection_names,
                        previous_aggregation_queries=previous_aggregations,
                        num_base_lm_examples=6,
                        return_example_uuids=True,
                    )
                )
            else:
                aggregation = await aggregate_generator.aforward(
                    lm=complex_lm,
                    available_collections=collection_names,
                    previous_aggregation_queries=previous_aggregations,
                )
        except Exception as e:
            yield Error(error_message=str(e))
            return

        if self.logger and aggregation.aggregation_queries is not None:
            self.logger.debug(f"Aggregation: {aggregation.aggregation_queries}")

        # Yield results to front end
        yield Response(text=aggregation.message_update)

        if tree_data.settings.USE_FEEDBACK:
            yield FewShotExamples(uuids=example_uuids)

        # Return if model deems aggregation impossible
        if aggregation.impossible or aggregation.aggregation_queries is None:
            if self.logger:
                self.logger.info(
                    "Model judged this aggregation to be impossible. Returning to the decision tree..."
                )
            for collection_name in collection_names:
                metadata = {
                    "collection_name": collection_name,
                    "impossible": tree_data.user_prompt,
                    "impossible_reasoning": (
                        aggregation.reasoning
                        if tree_data.settings.COMPLEX_USE_REASONING
                        else ""
                    ),
                    "aggregation_output": (
                        aggregation.aggregation_queries.model_dump()
                        if aggregation.aggregation_queries is not None
                        else None
                    ),
                }
                yield Aggregation([], metadata)

            return

        # Go through each response
        for i, aggregation_output in enumerate(aggregation.aggregation_queries):

            aggregation_output.target_collections = self._fix_collection_names(
                aggregation_output.target_collections,
                schemas,
            )

            # Execute query within Weaviate
            async with client_manager.connect_to_async_client() as client:
                try:
                    responses, code_strings = await execute_weaviate_aggregation(
                        client,
                        aggregation_output,
                        property_types={
                            collection_name: {
                                schemas[collection_name]["fields"][i]["name"]: schemas[
                                    collection_name
                                ]["fields"][i]["type"]
                                for i in range(len(schemas[collection_name]["fields"]))
                            }
                            for collection_name in collection_names
                        },
                        vectorised_by_collection=vectorised_by_collection,
                        schema=schemas,
                    )
                except QueryError as e:
                    yield Error(feedback=str(e))
                    continue
                except Exception as e:
                    for collection_name in aggregation_output.target_collections:
                        if self.logger:
                            self.logger.exception(
                                f"Error executing aggregation for {collection_name}: {str(e)}"
                            )
                        yield Error(error_message=str(e))
                    continue  # don't exit, try again for next aggregation output (might not error)

            yield Status(
                f"Aggregated over {', '.join(aggregation_output.target_collections)}"
            )

            for k, (collection_name, response) in enumerate(
                zip(aggregation_output.target_collections, responses)
            ):
                if self.logger and self.logger.level <= 20:
                    print(
                        Panel.fit(
                            code_strings[k],
                            title=f"{collection_name} (Weaviate Aggregation Query)",
                            border_style="yellow",
                            padding=(1, 1),
                        )
                    )

                objects = [
                    {
                        "num_items": (
                            response.total_count
                            if "total_count" in dir(response)
                            else None
                        ),
                        "collections": [
                            {collection_name: format_aggregation_response(response)}
                        ],
                    }
                ]
                metadata = {
                    "collection_name": collection_name,
                    "aggregation_output": aggregation_output.model_dump(),
                    "code": {
                        "language": "python",
                        "title": "Aggregation",
                        "text": code_strings[k],
                    },
                }

                yield Aggregation(objects, metadata)

        # if successful, yield training update
        yield TrainingUpdate(
            module_name="aggregate",
            inputs={
                **tree_data.to_json(),
                "available_collections": collection_names,
                "previous_aggregation_queries": previous_aggregations,
            },
            outputs=aggregation.__dict__["_store"],
        )
        if self.logger:
            self.logger.debug("Aggregation Tool finished!")
