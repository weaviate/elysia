from typing import AsyncGenerator, Union
from logging import Logger
from pydantic import BaseModel, Field

from rich import print
from rich.panel import Panel

import dspy

from elysia.util.elysia_chain_of_thought import ElysiaChainOfThought
from elysia.objects import Response, Status, Tool, Error, Result, Return, Retrieval
from elysia.tools.retrieval.chunk import AsyncCollectionChunker
from elysia.tools.retrieval.objects import (
    ConversationRetrieval,
    DocumentRetrieval,
    MessageRetrieval,
)
from elysia.tools.retrieval.prompt_templates import (
    QueryCreatorPrompt,
    construct_query_output_prompt,
)
from elysia.tools.retrieval.util import (
    execute_weaviate_query,
    QueryError,
    QueryOutput,
    NonVectorisedQueryOutput,
)
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
from elysia.util.objects import TrainingUpdate, TreeUpdate, FewShotExamples
from elysia.util.return_types import all_return_types


class Query(Tool):
    """
    Elysia Query Tool
    Queries the knowledge base with Weaviate using hybrid, semantic, or keyword search, applying filters and more.
    These are dynamically decided by an LLM, based on the user prompt and the collections available.

    Additionally, this Tool can perform the following actions:
    - Chunk collections that are too large to vectorise effectively
    - Dynamically return the retrieved objects to the frontend according to the display type decided by the LLM
    - Set a flag to summarise retrieved objects in the tree

    """

    def __init__(
        self,
        logger: Logger | None = None,
        summariser_in_tree: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            name="query",
            description="""
            Retrieves and displays specific data entries from the collections.
            Then, query with semantic search, keyword search, or a combination of both.
            Queries can be filtered, sorted, and more.
            Retrieving and displaying specific data entries rather than performing calculations or summaries.
            Do not use 'query' as a preliminary filtering step when 'aggregate' can achieve the same result more efficiently (if 'aggregate' is available).
            """,
            status="Querying...",
            inputs={
                "collection_names": {
                    "description": "the names of the collections that are most relevant to the query. If you are unsure, give all collections.",
                    "type": list[str],
                    "default": [],
                },
            },
            end=False,
        )

        self.logger = logger
        self.summariser_in_tree = summariser_in_tree
        self.retrieval_map = {
            "conversation": ConversationRetrieval,
            "message": MessageRetrieval,
            "document": DocumentRetrieval,
        }

    async def is_tool_available(
        self,
        tree_data: TreeData,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
    ) -> bool:
        """
        Only available if:
        1. There is a Weaviate connection
        2. There are collections available
        If this tool is not available, inform the user that they should make sure they have set the WCD_URL and WCD_API_KEY in the settings.
        And also they should make sure they have added collections to the tree.
        """
        return client_manager.is_client and tree_data.collection_names != []

    def _find_previous_queries(
        self, environment: dict, collection_names: list[str]
    ) -> dict:
        previous_queries = {}
        # create a dict of collection_name: previous_queries
        for collection_name in collection_names:
            # check if a query has happened at all for this collection
            if "query" in environment and collection_name in environment["query"]:
                for query in environment["query"][collection_name]:
                    if "query_output" in query["metadata"]:
                        if collection_name in previous_queries:
                            previous_queries[collection_name].append(
                                query["metadata"]["query_output"]
                            )
                        else:
                            previous_queries[collection_name] = [
                                query["metadata"]["query_output"]
                            ]

        return previous_queries

    def _evaluate_content_field(
        self, metadata_fields: list[dict]
    ) -> tuple[str | None, float | None]:
        """
        Find the largest text field in the metadata.
        Returns the field name with the highest mean value among text fields.
        """
        # Filter for text fields only
        text_fields = {
            field["name"]: field
            for field in metadata_fields
            if (field.get("type") == "text" and field.get("mean") is not None)
        }

        if not text_fields:
            return None, None

        # Find the text field with the largest mean value
        max_field = max(text_fields, key=lambda x: text_fields[x].get("mean", 0))
        return max_field, text_fields[max_field]["mean"]

    def _evaluate_needs_chunking(
        self,
        display_type: str,
        query_type: str,
        schema: dict,
        threshold: int = 400,  # number of tokens to be considered large enough to chunk
    ) -> bool:
        content_field, content_len = self._evaluate_content_field(
            schema["fields"],
        )

        return (
            content_field is not None
            and content_len > threshold
            and query_type != "filter_only"
            and display_type
            == "document"  # for the moment, the only type we chunk is document
        )

    def _fix_collection_names(
        self, collection_names: list[str], schemas: dict
    ) -> list[str]:
        return [
            correct_collection_name
            for correct_collection_name in list(schemas.keys())
            if correct_collection_name.lower()
            in [collection_name.lower() for collection_name in collection_names]
        ]

    def _fix_collection_names_in_dict(
        self, collection_dict: dict, schemas: dict
    ) -> dict:
        true_collection_names = list(schemas.keys())
        truth_map = {k.lower(): k for k in true_collection_names}
        return {truth_map.get(k.lower(), k): v for k, v in collection_dict.items()}

    def _parse_weaviate_error(self, error_message: str) -> str:
        if "no api key found" in error_message.lower():
            api_error_message = error_message[
                (
                    error_message.find("environment variable under ")
                    + len("environment variable under ")
                ) :
            ]
            api_error_message = api_error_message[: api_error_message.find('"\n')]
            return (
                f"The following API key is needed: {api_error_message} for this collection. "
                "The user must set the API key in the settings. "
                "Alternatively, you can try keyword (BM25) search instead."
            )

        return error_message

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
        **kwargs,
    ) -> AsyncGenerator[Return | TreeUpdate | Error | TrainingUpdate, None]:
        """
        Can perform three main functions:
        1. Query the knowledge base with Weaviate using hybrid, semantic, or keyword search, applying filters and more.
        2. Automatically chunk collections that are too large to vectorise effectively
        3. Summarise retrieved objects if requested
        Chunking creates a parallel quantized collection of chunks with cross references to the original collection.
        When filtering on the parallel chunked collection, filters are applied to the original collection by cross referencing.
        """

        if self.logger:
            self.logger.debug("Query Tool called!")
            self.logger.debug(f"Inputs: {inputs}")

        # Get the collection schemas (truth)
        schemas = tree_data.output_collection_metadata(with_mappings=True)

        # Find matching collection names (to match case sensitivity)
        if inputs["collection_names"] == [] or inputs["collection_names"] is None:
            collection_names = list(schemas.keys())
        else:
            collection_names = self._fix_collection_names(
                inputs["collection_names"], schemas
            )

        # Subset the schemas to just the collection names
        schemas = {
            collection_name: schemas[collection_name]
            for collection_name in collection_names
        }
        if self.logger:
            self.logger.debug(f"Collection names received: {collection_names}")

        # Set up the dspy model
        query_creator_prompt = QueryCreatorPrompt

        # check if collections are vectorised
        vectorised = any(
            (
                schemas[collection_name]["vectorizer"] is not None
                or (
                    schemas[collection_name]["named_vectors"] is not None
                    and schemas[collection_name]["named_vectors"] != []
                )
            )
            for collection_name in collection_names
        )

        if vectorised:
            query_creator_prompt = query_creator_prompt.append(
                name="query_outputs",
                type_=Union[list[QueryOutput], None],
                field=dspy.OutputField(
                    desc=construct_query_output_prompt(True),
                ),
            )
        else:
            query_creator_prompt = query_creator_prompt.append(
                name="query_outputs",
                type_=Union[list[NonVectorisedQueryOutput], None],
                field=dspy.OutputField(
                    desc=construct_query_output_prompt(False),
                ),
            )

        query_generator = ElysiaChainOfThought(
            query_creator_prompt,
            tree_data=tree_data,
            environment=True,
            collection_schemas=True,
            tasks_completed=True,
            message_update=True,
            collection_names=collection_names,
        )

        # Get some metadata about the collection
        previous_queries = self._find_previous_queries(
            tree_data.environment.environment,
            tree_data.collection_data.collection_names,
        )

        if self.logger:
            self.logger.debug(f"Previous queries: {previous_queries}")

        # get display types
        display_types = tree_data.output_collection_return_types()
        display_types = {
            collection_name: display_types[collection_name]
            for collection_name in collection_names
        }  # subset to just our collection names

        # get display type descriptions
        display_type_descriptions = {
            display_type: all_return_types[display_type]
            for collection_name in collection_names
            for display_type in display_types[collection_name]
        }

        # get searchable fields
        searchable_fields = {
            collection_name: [
                named_vector["name"]
                for named_vector in schemas[collection_name]["named_vectors"]
                if named_vector["enabled"]
            ]
            for collection_name in collection_names
            if schemas[collection_name]["named_vectors"] is not None
        }

        # Generate query with LLM
        try:
            if tree_data.settings.USE_FEEDBACK:
                query, example_uuids = (
                    await query_generator.aforward_with_feedback_examples(
                        feedback_model="query",
                        client_manager=client_manager,
                        base_lm=base_lm,
                        complex_lm=complex_lm,
                        available_collections=collection_names,
                        previous_queries=previous_queries,
                        collection_display_types=display_types,
                        display_type_descriptions=display_type_descriptions,
                        searchable_fields=searchable_fields,
                        num_base_lm_examples=6,
                        return_example_uuids=True,
                    )
                )
            else:
                query = await query_generator.aforward(
                    lm=complex_lm,
                    available_collections=collection_names,
                    previous_queries=previous_queries,
                    collection_display_types=display_types,
                    display_type_descriptions=display_type_descriptions,
                    searchable_fields=searchable_fields,
                )

        except Exception as e:
            yield Error(error_message=str(e))
            return

        if self.logger and query.query_outputs is not None:
            self.logger.debug(f"Query: {query.query_outputs}")
            self.logger.debug(f"Fields to search: {query.fields_to_search}")
            self.logger.debug(f"Data display: {query.data_display}")

        # Yield results to front end
        yield Response(text=query.message_update)
        if tree_data.settings.USE_FEEDBACK:
            yield FewShotExamples(uuids=example_uuids)

        # Return if model deems query impossible
        if (
            query.impossible
            or query.query_outputs is None
            or all(q is None for q in query.query_outputs)
            or len(query.query_outputs) == 0
        ):

            for collection_name in collection_names:
                metadata = {
                    "collection_name": collection_name,
                    "impossible": tree_data.user_prompt,
                    "impossible_reasoning": (
                        query.reasoning
                        if tree_data.settings.COMPLEX_USE_REASONING
                        else ""
                    ),
                    "query_output": (
                        query.query_outputs.model_dump()
                        if query.query_outputs is not None
                        else None
                    ),
                }
                yield Retrieval([], metadata)

            if self.logger:
                self.logger.warning(
                    f"Model judged query to be impossible. Returning to the decision tree..."
                )
            return

        # extract and error handle the query output
        for current_query_output in query.query_outputs:

            current_query_output.target_collections = self._fix_collection_names(
                current_query_output.target_collections, schemas
            )

            for collection_name in current_query_output.target_collections:
                if collection_name not in collection_names:
                    yield Error(
                        feedback=(
                            f"Collection {collection_name} in target_collections field of the query output, "
                            "but not found in the available collections. "
                            "Make sure you are using the correct collection names."
                        ),
                    )
                    return

        query.data_display = self._fix_collection_names_in_dict(
            query.data_display, schemas
        )
        query.fields_to_search = self._fix_collection_names_in_dict(
            query.fields_to_search, schemas
        )

        for collection_name in query.data_display:

            if collection_name not in collection_names:
                yield Error(
                    feedback=(
                        f"Collection {collection_name} in data_display keys, "
                        "but not found in the available collections. "
                        "Make sure you are using the correct collection names."
                    ),
                )
                return

        for collection_name in query.fields_to_search:
            if collection_name not in collection_names:
                yield Error(
                    feedback=(
                        f"Collection {collection_name} in fields_to_search keys, "
                        "but not found in the available collections. "
                        "Make sure you are using the correct collection names."
                    ),
                )
                return

        # if the fields_to_search is non-empty but there are no named vectors, ignore this and reset it
        for collection_name in query.fields_to_search:
            if query.fields_to_search[collection_name] is not None and (
                schemas[collection_name]["named_vectors"] is None
                or searchable_fields[collection_name] == []
            ):
                query.fields_to_search[collection_name] = None

        # Go through outputs for each unique query
        for i, query_output in enumerate(query.query_outputs):
            collection_names = query_output.target_collections.copy()

            for collection_name in collection_names:

                display_type = query.data_display[collection_name].display_type

                # Evaluate if this collection/query needs chunking
                needs_chunking = self._evaluate_needs_chunking(
                    display_type,
                    query_output.search_type,
                    schemas[collection_name],
                )
                content_field, _ = self._evaluate_content_field(
                    schemas[collection_name]["fields"],
                )  # used for chunking

                if needs_chunking and content_field is not None:

                    if self.logger:
                        self.logger.debug(f"Chunking {collection_name}")

                    # set up chunking (create reference in this collection)
                    collection_chunker = AsyncCollectionChunker(collection_name)
                    await collection_chunker.create_chunked_reference(
                        content_field, client_manager
                    )

                    # create a copy of the query output
                    query_output_copy = query_output.model_copy(deep=True)

                    # update the limit to be larger
                    query_output_copy.limit = query_output.limit * 3

                    # update the collection to be the unchunked collection ONLY
                    query_output_copy.target_collections = [collection_name]

                    # run this augmented query for the unchunked collection
                    async with client_manager.connect_to_async_client() as client:
                        try:
                            unchunked_response, _ = await execute_weaviate_query(
                                client,
                                query_output_copy,
                                reference_property="isChunked",
                                named_vector_fields=query.fields_to_search,
                                property_types={
                                    collection_name: {field["name"]: field["type"]}
                                    for field in schemas[collection_name]["fields"]
                                },
                                schema=schemas,
                            )
                        except QueryError as e:
                            yield Error(feedback=str(e))
                            continue
                        except Exception as e:
                            yield Error(error_message=str(e))
                            continue

                    # chunk the unchunked response
                    yield Status(
                        f"Chunking {len(unchunked_response[0].objects)} objects in {collection_name}..."
                    )
                    await collection_chunker(
                        unchunked_response[0], content_field, client_manager
                    )

                    # modify the original query output to be the chunked collection
                    for j, c in enumerate(query_output.target_collections):
                        if c == collection_name:
                            query_output.target_collections[j] = (
                                f"ELYSIA_CHUNKED_{collection_name.lower()}__"
                            )
                    query.fields_to_search[
                        f"ELYSIA_CHUNKED_{collection_name.lower()}__"
                    ] = None

                    if self.logger:
                        self.logger.debug(
                            f"Chunked {collection_name} and updated query output to {query_output.target_collections}"
                        )

            # Execute query within Weaviate
            async with client_manager.connect_to_async_client() as client:
                try:
                    responses, code_strings = await execute_weaviate_query(
                        client,
                        query_output,
                        named_vector_fields=query.fields_to_search,
                        property_types={
                            collection_name: {
                                field["name"]: field["type"]
                                for field in schemas[collection_name]["fields"]
                            }
                            for collection_name in collection_names
                        },
                        schema=schemas,
                    )
                except QueryError as e:
                    yield Error(feedback=str(e))
                    continue
                except Exception as e:
                    for collection_name in collection_names:
                        yield Error(error_message=str(e))
                    continue

                yield Status(
                    f"Retrieved {sum(len(x.objects) for x in responses)} objects from {len(collection_names)} collections..."
                )

            for k, (collection_name, response) in enumerate(
                zip(collection_names, responses)
            ):
                if self.logger and self.logger.level <= 20:
                    print(
                        Panel.fit(
                            code_strings[k],
                            title=f"{collection_name} (Weaviate Query)",
                            border_style="yellow",
                            padding=(1, 1),
                        )
                    )

                # Get display type and summarise items bool
                display_type = query.data_display[collection_name].display_type
                summarise_items = query.data_display[collection_name].summarise_items

                # get query output formatted
                query_output_formatted = query_output.model_dump()
                query_output_formatted["target_collections"] = collection_names

                # Get the objects from the response
                objects = []
                for obj in response.objects:
                    objects.append({k: v for k, v in obj.properties.items()})
                    objects[-1]["uuid"] = str(obj.uuid)

                # Write various metadata for LLM parsing
                metadata = {
                    "collection_name": collection_name,
                    "display_type": display_type,
                    "needs_summarising": summarise_items,
                    "query_text": query_output.search_query,
                    "query_type": query_output.search_type,
                    "chunked": self._evaluate_needs_chunking(
                        display_type,
                        query_output.search_type,
                        schemas[collection_name],
                    ),
                    "query_output": query_output_formatted,
                    "code": {
                        "language": "python",
                        "title": "Query",
                        "text": code_strings[k],
                    },
                }

                # Create the retrieval object
                if display_type in self.retrieval_map:
                    output = self.retrieval_map[display_type](
                        objects,
                        metadata,
                        mapping=(
                            schemas[collection_name]["mappings"][display_type]
                            if display_type != "table"
                            else None
                        ),
                    )
                else:
                    output = Retrieval(
                        objects,
                        metadata,
                        payload_type=display_type,
                        name=collection_name,
                        mapping=(
                            schemas[collection_name]["mappings"][display_type]
                            if display_type != "table"
                            else None
                        ),
                    )

                # If the display type is document or conversation, initialise the object (requires some async operations)
                if isinstance(output, DocumentRetrieval) or isinstance(
                    output, ConversationRetrieval
                ):
                    await output.async_init(client_manager)

                if summarise_items and self.summariser_in_tree:
                    if (
                        "items_to_summarise"
                        not in tree_data.environment.hidden_environment
                    ):
                        tree_data.environment.hidden_environment[
                            "items_to_summarise"
                        ] = []
                    tree_data.environment.hidden_environment[
                        "items_to_summarise"
                    ].append(output)
                else:
                    yield output

        # if successful, yield training update
        yield TrainingUpdate(
            module_name="query",
            inputs={
                "available_collections": collection_names,
                "previous_queries": previous_queries,
                "collection_display_types": display_types,
                "searchable_fields": searchable_fields,
                **tree_data.to_json(),
            },
            outputs=query.__dict__["_store"],
        )
        if self.logger:
            self.logger.debug("Query Tool finished!")
