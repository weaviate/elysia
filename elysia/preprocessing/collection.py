import random
import dspy
from rich.progress import Progress
from typing import AsyncGenerator

# Weaviate
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Metrics, Filter
from weaviate.collections import CollectionAsync
from weaviate.classes.config import Configure, Property, DataType, Tokenization

from elysia.config import nlp, Settings, load_base_lm, ElysiaKeyManager
from elysia.config import settings as environment_settings

from elysia.util import return_types as rt

from elysia.preprocessing.prompt_templates import (
    CollectionSummariserPrompt,
    DataMappingPrompt,
    ReturnTypePrompt,
    PromptSuggestorPrompt,
)
from elysia.util.collection import async_get_collection_data_types
from elysia.util.async_util import asyncio_run
from elysia.util.parsing import format_dict_to_serialisable
from elysia.util.client import ClientManager


class ProcessUpdate:
    def __init__(self, collection_name: str, total: int) -> None:
        self.collection_name = collection_name
        self.total = total
        self.progress = 0

    def update_progress(self) -> None:
        self.progress += 1

    def update_total(self, total: int) -> None:
        self.total = total

    async def to_frontend(
        self, completed: bool = False, message: str = "", error: str = "", **kwargs
    ) -> dict:
        return {
            "type": ("update" if not completed else "completed"),
            "collection_name": self.collection_name,
            "progress": min(self.progress / self.total, 0.99) if not completed else 1.0,
            "message": message,
            "error": error,
        }

    async def __call__(self, *args, **kwargs) -> dict:
        self.update_progress()
        return await self.to_frontend(*args, **kwargs)


async def _summarise_collection(
    collection_summariser_prompt: dspy.Module,
    properties: dict,
    subset_objects: list[dict],
    len_collection: int,
    settings: Settings,
    lm: dspy.LM,
) -> tuple[str, dict]:

    with ElysiaKeyManager(settings):
        prediction = await collection_summariser_prompt.aforward(
            data_sample=subset_objects,
            data_fields=list(properties.keys()),
            sample_size=f"""
            {len(subset_objects)} out of a total of {len_collection}.
            You are seeing {round(len(subset_objects)/len_collection*100, 2)}% of the objects.
            """,
            lm=lm,
        )

    summary_concat = ""
    for sentence in [
        prediction.overall_summary,
        prediction.relationships,
        prediction.structure,
        prediction.irregularities,
    ]:
        if (
            sentence.endswith(".")
            or sentence.endswith("?")
            or sentence.endswith("!")
            or sentence.endswith("\n")
        ):
            summary_concat += f"{sentence} "
        else:
            summary_concat += f"{sentence}."

    return summary_concat, prediction.field_descriptions


async def _evaluate_field_statistics(
    collection: CollectionAsync,
    properties: dict,
    property: str,
    len_collection: int,
    sample_objects: list[dict],
) -> dict:
    out = {}
    out["type"] = properties[property] if properties[property] != "number" else "float"
    out["name"] = property
    out["range"] = None
    out["mean"] = None
    out["date_range"] = None
    out["date_median"] = None
    out["groups"] = None

    if not out["type"].startswith("object"):

        groups_response = await collection.aggregate.over_all(
            total_count=True, group_by=GroupByAggregate(prop=property, limit=30)
        )

        if len(groups_response.groups) > 15:
            groups = [
                {
                    "value": str(
                        group.grouped_by.value
                    ),  # must force to string for weaviate
                    "count": group.total_count,
                }
                for group in groups_response.groups
                if group.total_count > 1
            ]
        else:
            groups = [
                {
                    "value": str(group.grouped_by.value),
                    "count": group.total_count,
                }
                for group in groups_response.groups
            ]

        total_count = sum(group["count"] for group in groups)
        remainder = len_collection - total_count
        group_coverage = total_count / len_collection
        if remainder > 0:
            groups.append(
                {
                    "value": "<undefined_group> (not used for filtering)",
                    "count": remainder,
                }
            )

        # check if groups are useful
        if len(groups) == 1 or group_coverage < 0.5:
            out["groups"] = None
        else:
            out["groups"] = groups

    # Number (summary statistics)
    if properties[property] == "int":
        response = await collection.aggregate.over_all(
            return_metrics=[
                Metrics(property).integer(mean=True, maximum=True, minimum=True),
            ]
        )
        out["range"] = [
            response.properties[property].minimum,
            response.properties[property].maximum,
        ]
        out["mean"] = response.properties[property].mean
        out["date_range"] = None
        out["date_median"] = None

    elif properties[property] == "number":
        response = await collection.aggregate.over_all(
            return_metrics=[
                Metrics(property).number(mean=True, maximum=True, minimum=True),
            ]
        )
        out["range"] = [
            response.properties[property].minimum,
            response.properties[property].maximum,
        ]
        out["mean"] = response.properties[property].mean
        out["date_range"] = None
        out["date_median"] = None

    # Text (lengths)
    elif properties[property] == "text":

        # For text, we want to evaluate the length of the text in tokens (use spacy)
        lengths = []
        for obj in sample_objects:
            if property in obj and isinstance(obj[property], str):
                lengths.append(len(nlp(obj[property])))

        if len(lengths) == 0:
            out["range"] = None
            out["mean"] = None
        else:
            out["range"] = [min(lengths), max(lengths)]
            out["mean"] = sum(lengths) / len(lengths)
        out["date_median"] = None
        out["date_range"] = None

    # Boolean (grouping + mean)
    elif properties[property] == "boolean":
        response = await collection.aggregate.over_all(
            return_metrics=[Metrics(property).boolean(percentage_true=True)]
        )
        out["mean"] = response.properties[property].percentage_true
        out["range"] = [0, 1]
        out["date_range"] = None
        out["date_median"] = None

    # Date (summary statistics)
    elif properties[property] == "date":
        response = await collection.aggregate.over_all(
            return_metrics=[
                Metrics(property).date_(median=True, minimum=True, maximum=True)
            ]
        )
        out["date_range"] = [
            response.properties[property].minimum,
            response.properties[property].maximum,
        ]
        out["mean"] = None
        out["date_median"] = response.properties[property].median
        out["range"] = None
        out["mean"] = None

    # List (lengths)
    elif properties[property].endswith("[]"):
        lengths = [len(obj[property]) for obj in sample_objects]

        if len(lengths) == 0:
            out["range"] = None
            out["mean"] = None
        else:
            out["range"] = [min(lengths), max(lengths)]
            out["mean"] = sum(lengths) / len(lengths)
        out["date_range"] = None
        out["date_median"] = None

    return out


async def _suggest_prompts(
    prompt_suggestor_prompt: dspy.Module,
    collection_information: dict,
    example_objects: list[dict],
    settings: Settings,
    lm: dspy.LM,
) -> list[str]:
    with ElysiaKeyManager(settings):
        prediction = await prompt_suggestor_prompt.aforward(
            collection_information=collection_information,
            example_objects=example_objects,
            lm=lm,
        )
    return prediction.prompt_suggestions


async def _evaluate_return_types(
    return_type_prompt: dspy.Module,
    collection_summary: str,
    data_fields: dict,
    example_objects: list[dict],
    settings: Settings,
    lm: dspy.LM,
) -> list[str]:

    with ElysiaKeyManager(settings):
        prediction = await return_type_prompt.aforward(
            collection_summary=collection_summary,
            data_fields=data_fields,
            example_objects=example_objects,
            possible_return_types=rt.specific_return_types,
            lm=lm,
        )
    return_types = prediction.return_types

    if return_types == []:
        return_types = ["generic"]

    # if conversation is selected, then message should also be selected
    if "conversation" in return_types and "message" not in return_types:
        return_types.append("message")

    return return_types


async def _define_mappings(
    data_mapping_prompt: dspy.Module,
    input_fields: list,
    output_fields: list,
    properties: dict,
    collection_information: dict,
    example_objects: list[dict],
    mapping_type: str,
    settings: Settings,
    lm: dspy.LM,
) -> dict:

    with ElysiaKeyManager(settings):
        prediction = await data_mapping_prompt.aforward(
            mapping_type=mapping_type,
            input_data_fields=input_fields,
            output_data_fields=output_fields,
            input_data_types=properties,
            collection_information=collection_information,
            example_objects=example_objects,
            lm=lm,
        )

    return prediction.field_mapping


async def _evaluate_index_properties(collection: CollectionAsync) -> dict:
    schema_info = await collection.config.get()

    index_properties = {
        "isNullIndexed": schema_info.inverted_index_config.index_null_state,
        "isLengthIndexed": schema_info.inverted_index_config.index_property_length,
        "isTimestampIndexed": schema_info.inverted_index_config.index_timestamps,
    }
    return index_properties


async def _find_vectorisers(collection: CollectionAsync) -> dict[str, dict]:
    schema_info = await collection.config.get()
    if not schema_info.vector_config:
        named_vectors = None
    else:
        named_vectors = [
            {
                "name": vector,
                "vectorizer": schema_info.vector_config[
                    vector
                ].vectorizer.vectorizer.name,
                "model": (
                    schema_info.vector_config[vector].vectorizer.model["model"]
                    if "model" in schema_info.vector_config[vector].vectorizer.model
                    else None
                ),
                "source_properties": schema_info.vector_config[
                    vector
                ].vectorizer.source_properties,
                "enabled": True,
                "description": "",
            }
            for vector in schema_info.vector_config
        ]

    if not schema_info.vectorizer_config:
        vectoriser = None
    else:
        vectoriser = {
            "vectorizer": schema_info.vectorizer_config.vectorizer.name,
            "model": (
                schema_info.vectorizer_config.model["model"]
                if "model" in schema_info.vectorizer_config.model
                else None
            ),
        }

    return named_vectors, vectoriser


async def preprocess_async(
    collection_name: str,
    client_manager: ClientManager | None = None,
    min_sample_size: int = 10,
    max_sample_size: int | None = None,
    num_sample_tokens: int = 30000,
    force: bool = False,
    percentage_correct_threshold: float = 0.3,
    settings: Settings = environment_settings,
) -> AsyncGenerator[dict, None]:
    """
    Preprocess a collection, obtain a LLM-generated summary of the collection,
    a set of statistics for each field (such as unique categories), and a set of mappings
    from the fields to the frontend-specific fields in Elysia.

    In order:

    1. Evaluate all the data fields and groups/statistics of the data fields as a whole
    2. Write a summary of the collection via an LLM
    3. Evaluate what return types are available for this collection
    4. For each data field in the collection, evaluate what corresponding entry goes to what field in the return type (mapping)
    5. Save as a ELYSIA_METADATA__ collection

    Args:
        collection_name (str): The name of the collection to preprocess.
        client_manager (ClientManager): The client manager to use.
        min_sample_size (int): The minimum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 10.
        max_sample_size (int): The maximum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 20.
        num_sample_tokens (int): The number of tokens to approximately sample from the collection to evaluate the summary.
            The preprocessor will aim to use this many tokens in the sample objects to evaluate the summary.
            But will not exceed the maximum number of objects specified by `max_sample_size`, and always use at least `min_sample_size` objects.
        force (bool): Whether to force the preprocessor to run even if the collection already exists. Optional, defaults to False.
        threshold_for_missing_fields (float): The threshold for the number of missing fields in the data mapping. Optional, defaults to 0.1.
        settings (Settings): The settings to use. Optional, defaults to the environment variables/configured settings.

    Returns:
        AsyncGenerator[dict, None]: A generator that yields dictionaries with the status updates and progress of the preprocessor.
    """

    collection_summariser_prompt = dspy.ChainOfThought(CollectionSummariserPrompt)
    return_type_prompt = dspy.ChainOfThought(ReturnTypePrompt)
    data_mapping_prompt = dspy.ChainOfThought(DataMappingPrompt)
    prompt_suggestor_prompt = dspy.ChainOfThought(PromptSuggestorPrompt)

    lm = load_base_lm(settings)
    logger = settings.logger
    process_update = ProcessUpdate(collection_name, len(rt.specific_return_types) + 5)

    if client_manager is None:
        client_manager = ClientManager(
            wcd_url=settings.WCD_URL, wcd_api_key=settings.WCD_API_KEY
        )
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    try:
        # Check if the collection exists
        async with client_manager.connect_to_async_client() as client:
            if not await client.collections.exists(collection_name):
                raise Exception(f"Collection {collection_name} does not exist!")

        # Check if the preprocessed collection exists
        if (
            await preprocessed_collection_exists_async(collection_name, client_manager)
            and not force
        ):
            logger.info(f"Preprocessed metadata for {collection_name} already exists!")
            return

        # Get the collection and its properties
        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get(collection_name)
            properties = await async_get_collection_data_types(client, collection_name)

        # get number of items in collection
        agg = await collection.aggregate.over_all(total_count=True)
        len_collection: int = agg.total_count  # type: ignore

        if max_sample_size is None and len_collection > 50_000:
            max_sample_size = 20
            logger.warning(
                f"Collection is large (greater than 50,000 objects), causing slowdown in pre-processing. "
                f"Reducing maximum sample size to {max_sample_size} objects. "
                "To override this, set `max_sample_size` as an argument to preprocess."
            )
        elif max_sample_size is None:
            max_sample_size = 50

        # Randomly sample sample_size objects for the summary
        indices = random.sample(
            range(min(99_999, len_collection)),
            max(min(max_sample_size, len_collection), 1),
        )

        # Get first object to estimate token count
        obj = await collection.query.fetch_objects(limit=1, offset=indices[0])
        token_count_0 = len(nlp(str(obj.objects[0].properties)))
        subset_objects: list[dict] = [obj.objects[0].properties]  # type: ignore

        # Get number of objects to sample to get close to num_sample_tokens
        num_sample_objects = min(
            max(min_sample_size, num_sample_tokens // token_count_0),
            max_sample_size,
        )

        # Estimate number of tokens
        logger.debug(
            f"Estimated token count of sample: {token_count_0*num_sample_objects}"
        )
        logger.debug(f"Number of objects in sample: {num_sample_objects}")

        for index in indices[1:num_sample_objects]:
            obj = await collection.query.fetch_objects(limit=1, offset=index)
            subset_objects.append(obj.objects[0].properties)  # type: ignore

        # Summarise the collection using LLM and the subset of the data
        summary, field_descriptions = await _summarise_collection(
            collection_summariser_prompt,
            properties,
            subset_objects,
            len_collection,
            settings,
            lm,
        )

        yield await process_update(
            message="Generated summary of collection",
        )

        if len_collection > 10_000:  # arbitrary cutoff for estimating field statistics
            full_response = subset_objects
        else:
            weaviate_resp = await collection.query.fetch_objects(limit=len_collection)
            full_response = [obj.properties for obj in weaviate_resp.objects]

        # Initialise the output
        named_vectors, vectoriser = await _find_vectorisers(collection)
        out = {
            "name": collection_name,
            "length": len_collection,
            "summary": summary,
            "index_properties": await _evaluate_index_properties(collection),
            "named_vectors": named_vectors,
            "vectorizer": vectoriser,
            "fields": [],
            "mappings": {},
        }

        # Evaluate the summary statistics of each field
        for property in properties:
            out["fields"].append(
                await _evaluate_field_statistics(
                    collection, properties, property, len_collection, full_response
                )
            )
            if property in field_descriptions:
                out["fields"][-1]["description"] = field_descriptions[property]
            else:
                out["fields"][-1]["description"] = ""

        yield await process_update(
            message="Evaluated field statistics",
        )

        return_types = await _evaluate_return_types(
            return_type_prompt,
            summary,
            properties,
            subset_objects,
            settings,
            lm,
        )

        yield await process_update(
            message="Evaluated return types",
        )
        process_update.update_total(len(return_types) + 5)

        # suggest prompts
        out["prompts"] = await _suggest_prompts(
            prompt_suggestor_prompt,
            out,
            subset_objects,
            settings,
            lm,
        )

        yield await process_update(
            message="Created suggestions for prompts",
        )

        # For each return type created above, define the mappings from the properties to the frontend types
        mappings = {}
        for return_type in return_types:
            fields = rt.types_dict[return_type]

            mapping = await _define_mappings(
                data_mapping_prompt,
                mapping_type=return_type,
                input_fields=list(fields.keys()),
                output_fields=list(properties.keys()),
                properties=properties,
                collection_information=out,
                example_objects=subset_objects,
                settings=settings,
                lm=lm,
            )

            # remove any extra fields the model may have added
            mapping = {k: v for k, v in mapping.items() if k in list(fields.keys())}

            yield await process_update(
                message=f"Defined mappings for {return_type}",
            )

            mappings[return_type] = mapping

        new_return_types = []
        for return_type in return_types:

            # check if the `conversation_id` field is in the mapping (required for conversation type)
            if return_type == "conversation" and (
                (
                    mappings[return_type]["conversation_id"] is None
                    or mappings[return_type]["conversation_id"] == ""
                )
                or (
                    mappings[return_type]["message_id"] is None
                    or mappings[return_type]["message_id"] == ""
                )
            ):
                continue

            # If less than threshold_for_missing_fields%, keep the return type
            num_missing = sum([m == "" for m in list(mappings[return_type].values())])
            perc_correct = 1 - (num_missing / len(mappings[return_type].keys()))
            if perc_correct >= percentage_correct_threshold:
                new_return_types.append(return_type)

        # If no return types are left, fall-back to generic
        if len(new_return_types) == 0:

            # Map for generic
            mapping = await _define_mappings(
                data_mapping_prompt,
                mapping_type="generic",
                input_fields=list(rt.generic.keys()),
                output_fields=list(properties.keys()),
                properties=properties,
                collection_information=out,
                example_objects=subset_objects,
                settings=settings,
                lm=lm,
            )
            yield await process_update(
                message="No display types found, defined mappings for generic",
            )

            # remove any extra fields the model may have added
            mapping = {k: v for k, v in mapping.items() if k in list(rt.generic.keys())}
            mappings["generic"] = mapping

            # re-check the threshold for missing fields on generic
            num_missing = sum([m == "" for m in list(mappings[return_type].values())])
            perc_correct = 1 - (num_missing / len(mappings[return_type].keys()))
            if perc_correct >= percentage_correct_threshold:
                new_return_types = ["generic"]

        # Add the mappings to the output
        out["mappings"] = {
            return_type: mappings[return_type] for return_type in new_return_types
        }

        # always include the table return type
        out["mappings"]["table"] = {field: field for field in properties.keys()}

        # Delete existing metadata if it exists
        if await preprocessed_collection_exists_async(collection_name, client_manager):
            await delete_preprocessed_collection_async(collection_name, client_manager)

        # Save final metadata to a collection
        async with client_manager.connect_to_async_client() as client:
            if await client.collections.exists(f"ELYSIA_METADATA__"):
                metadata_collection = client.collections.get("ELYSIA_METADATA__")
            else:
                metadata_collection = await client.collections.create(
                    f"ELYSIA_METADATA__",
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(
                            name="name",
                            data_type=DataType.TEXT,
                            tokenization=Tokenization.FIELD,
                        ),
                        Property(
                            name="length",
                            data_type=DataType.NUMBER,
                        ),
                        Property(
                            name="summary",
                            data_type=DataType.TEXT,
                        ),
                        Property(
                            name="index_properties",
                            data_type=DataType.OBJECT,
                            nested_properties=[
                                Property(
                                    name="isNullIndexed",
                                    data_type=DataType.BOOL,
                                ),
                                Property(
                                    name="isLengthIndexed",
                                    data_type=DataType.BOOL,
                                ),
                                Property(
                                    name="isTimestampIndexed",
                                    data_type=DataType.BOOL,
                                ),
                            ],
                        ),
                        Property(
                            name="named_vectors",
                            data_type=DataType.OBJECT_ARRAY,
                            nested_properties=[
                                Property(
                                    name="name",
                                    data_type=DataType.TEXT,
                                ),
                                Property(
                                    name="vectorizer",
                                    data_type=DataType.TEXT,
                                ),
                                Property(
                                    name="model",
                                    data_type=DataType.TEXT,
                                ),
                                Property(
                                    name="source_properties",
                                    data_type=DataType.TEXT_ARRAY,
                                ),
                                Property(
                                    name="enabled",
                                    data_type=DataType.BOOL,
                                ),
                                Property(
                                    name="description",
                                    data_type=DataType.TEXT,
                                ),
                            ],
                        ),
                        Property(
                            name="vectorizer",
                            data_type=DataType.OBJECT,
                            nested_properties=[
                                Property(name="vectorizer", data_type=DataType.TEXT),
                                Property(name="model", data_type=DataType.TEXT),
                            ],
                        ),
                        Property(
                            name="fields",
                            data_type=DataType.OBJECT_ARRAY,
                            nested_properties=[
                                Property(
                                    name="name",
                                    data_type=DataType.TEXT,
                                ),
                                Property(
                                    name="type",
                                    data_type=DataType.TEXT,
                                ),
                                Property(
                                    name="description",
                                    data_type=DataType.TEXT,
                                ),
                                Property(
                                    name="range",
                                    data_type=DataType.NUMBER_ARRAY,
                                ),
                                Property(
                                    name="date_range",
                                    data_type=DataType.DATE_ARRAY,
                                ),
                                Property(
                                    name="groups",
                                    data_type=DataType.OBJECT_ARRAY,
                                    nested_properties=[
                                        Property(
                                            name="value",
                                            data_type=DataType.TEXT,
                                        ),
                                        Property(name="count", data_type=DataType.INT),
                                    ],
                                ),
                                Property(
                                    name="date_median",
                                    data_type=DataType.DATE,
                                ),
                                Property(
                                    name="mean",
                                    data_type=DataType.NUMBER,
                                ),
                            ],
                        ),
                        # leave mappings for auto-schema generation
                    ],
                    inverted_index_config=Configure.inverted_index(
                        index_null_state=True,
                    ),
                )
            await metadata_collection.data.insert(out)

        yield await process_update(
            completed=True,
            message="Saved metadata to Weaviate",
        )

    except Exception as e:
        yield await process_update(
            error=f"Error preprocessing collection: {str(e)}",
        )

    finally:
        if close_clients_after_completion:
            await client_manager.close_clients()


async def _preprocess_async(
    collection_names: list[str] | str,
    client_manager: ClientManager | None = None,
    min_sample_size: int = 10,
    max_sample_size: int | None = None,
    num_sample_tokens: int = 30000,
    settings: Settings = environment_settings,
    force: bool = False,
) -> None:

    if client_manager is None:
        client_manager = ClientManager(
            wcd_url=settings.WCD_URL, wcd_api_key=settings.WCD_API_KEY
        )
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    if isinstance(collection_names, str):
        collection_names = [collection_names]

    try:

        if settings.LOGGING_LEVEL_INT > 20:
            for collection_name in collection_names:
                async for result in preprocess_async(
                    collection_name=collection_name,
                    client_manager=client_manager,
                    min_sample_size=min_sample_size,
                    max_sample_size=max_sample_size,
                    num_sample_tokens=num_sample_tokens,
                    force=force,
                    settings=settings,
                ):
                    if (
                        result is not None
                        and "error" in result
                        and result["error"] != ""
                    ):
                        raise Exception(result["error"])

        else:
            for collection_name in collection_names:
                with Progress() as progress:
                    task = progress.add_task(
                        f"Preprocessing {collection_name}", total=1
                    )
                    async for result in preprocess_async(
                        collection_name=collection_name,
                        client_manager=client_manager,
                        min_sample_size=min_sample_size,
                        max_sample_size=max_sample_size,
                        num_sample_tokens=num_sample_tokens,
                        force=force,
                        settings=settings,
                    ):
                        if (
                            result is not None
                            and "error" in result
                            and result["error"] != ""
                        ):
                            raise Exception(result["error"])
                        elif result is not None and "progress" in result:
                            progress.update(
                                task,
                                completed=result["progress"],
                                description=f"({collection_name}) {result['message']}",
                            )
                    progress.update(
                        task,
                        completed=1,
                        description=f"({collection_name}) Preprocessing complete!",
                    )

    finally:
        if close_clients_after_completion:
            await client_manager.close_clients()


def preprocess(
    collection_names: str | list[str],
    client_manager: ClientManager | None = None,
    min_sample_size: int = 10,
    max_sample_size: int | None = None,
    num_sample_tokens: int = 30000,
    settings: Settings = environment_settings,
    force: bool = False,
) -> None:
    """
    Preprocess a collection, obtain a LLM-generated summary of the collection,
    a set of statistics for each field (such as unique categories), and a set of mappings
    from the fields to the frontend-specific fields in Elysia.

    In order:
    1. Evaluate all the data fields and groups/statistics of the data fields as a whole
    2. Write a summary of the collection via an LLM
    3. Evaluate what return types are available for this collection
    4. For each data field in the collection, evaluate what corresponding entry goes to what field in the return type (mapping)
    5. Save as a ELYSIA_METADATA__ collection

    Depending on the size of objects in the collection, you can choose the minimum and maximum sample size,
    which will be used to create a sample of objects for the LLM to create a collection summary.
    If your objects are particularly large, you can set the sample size to be smaller, to use less tokens and speed up the LLM processing.
    If your objects are small, you can set the sample size to be larger, to get a more accurate summary.
    This is a trade-off between speed/compute and accuracy.

    But note that the pre-processing step only needs to be done once for each collection.
    The output of this function is cached, so that if you run it again, it will not re-process the collection (unless the force flag is set to True).

    This function saves the output into a collection called ELYSIA_METADATA__, which is automatically called by Elysia.
    This is saved to whatever Weaviate cluster URL/API key you have configured, or in your environment variables.
    You can change this by setting the `wcd_url` and `wcd_api_key` in the settings, and pass this Settings object to this function.

    Args:
        collection_names (str | list[str]): The name(s) of the collections to preprocess.
            Can supply either a single string for one collection, or a list of strings for multiple collections.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
        min_sample_size (int): The minimum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 10.
        max_sample_size (int): The maximum number of objects to sample from the collection to evaluate the statistics/summary. Optional, defaults to 20.
        num_sample_tokens (int): The maximum number of tokens in the sample objects used to evaluate the summary. Optional, defaults to 30000.
        settings (Settings): The settings to use. Optional, defaults to the environment variables/configured settings.
        force (bool): Whether to force the preprocessor to run even if the collection already exists. Optional, defaults to False.
    """

    asyncio_run(
        _preprocess_async(
            collection_names,
            client_manager,
            min_sample_size,
            max_sample_size,
            num_sample_tokens,
            settings,
            force,
        )
    )


async def preprocessed_collection_exists_async(
    collection_name: str, client_manager: ClientManager | None = None
) -> bool:
    """
    Async version of `preprocessed_collection_exists`.

    Args:
        collection_name (str): The name of the collection to check.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.

    Returns:
        bool: True if the collection exists, False otherwise.
    """
    if client_manager is None:
        client_manager = ClientManager()
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    async with client_manager.connect_to_async_client() as client:
        metadata_exists = await client.collections.exists(f"ELYSIA_METADATA__")
        if not metadata_exists:
            return False

        metadata_collection = client.collections.get("ELYSIA_METADATA__")
        metadata = await metadata_collection.query.fetch_objects(
            filters=Filter.by_property("name").equal(collection_name)
        )

    if close_clients_after_completion:
        await client_manager.close_clients()

    return metadata is not None and len(metadata.objects) > 0


def preprocessed_collection_exists(
    collection_name: str, client_manager: ClientManager | None = None
) -> bool:
    """
    Check if the preprocessed collection exists in the Weaviate cluster.
    This function simply checks if the cached preprocessed metadata exists in the Weaviate cluster.
    It does so by checking if the collection name exists in the ELYSIA_METADATA__ collection.

    Args:
        collection_name (str): The name of the collection to check.
        client_manager (ClientManager): The client manager to use.
    """
    return asyncio_run(
        preprocessed_collection_exists_async(collection_name, client_manager)
    )


async def delete_preprocessed_collection_async(
    collection_name: str, client_manager: ClientManager | None = None
) -> None:
    """
    Delete the preprocessed collection from the Weaviate cluster.
    This function simply deletes the cached preprocessed metadata from the Weaviate cluster.
    It does so by deleting the object in the collection ELYSIA_METADATA__ with the name of the collection.

    Args:
        collection_name (str): The name of the collection to delete the preprocessed metadata for.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
    """
    if client_manager is None:
        client_manager = ClientManager()
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    async with client_manager.connect_to_async_client() as client:
        if await client.collections.exists(f"ELYSIA_METADATA__"):
            metadata_collection = client.collections.get("ELYSIA_METADATA__")
            metadata = await metadata_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(collection_name),
                limit=1,
            )
            if metadata is not None and len(metadata.objects) > 0:
                await metadata_collection.data.delete_by_id(metadata.objects[0].uuid)
            else:
                raise Exception(f"Metadata for {collection_name} does not exist")

    if close_clients_after_completion:
        await client_manager.close_clients()


def delete_preprocessed_collection(
    collection_name: str, client_manager: ClientManager | None = None
) -> None:
    """
    Delete a preprocessed collection.
    This function allows you to delete the preprocessing done for a particular collection.
    It does so by deleting the object in the ELYSIA_METADATA__ collection with the name of the collection.

    Args:
        collection_name (str): The name of the collection to delete.
        client_manager (ClientManager): The client manager to use.
    """
    return asyncio_run(
        delete_preprocessed_collection_async(collection_name, client_manager)
    )


def edit_preprocessed_collection(
    collection_name: str,
    client_manager: ClientManager | None = None,
    named_vectors: list[dict] | None = None,
    summary: str | None = None,
    mappings: dict[str, dict[str, str]] | None = None,
    fields: list[dict[str, str] | None] | None = None,
) -> None:
    """
    Edit a preprocessed collection.
    This function allows you to edit the named vectors, summary, mappings, and fields of a preprocessed collection.
    It does so by updating the ELYSIA_METADATA__ collection.
    Find available mappings in the `elysia.util.return_types` module.

    Args:
        collection_name (str): The name of the collection to edit.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
        named_vectors (list[dict]): The named vectors to update. This has fields "name", "enabled", and "description".
            The "name" is used to identify the named vector to change (the name will not change).
            Set "enabled" to True/False to enable/disable the named vector.
            Set "description" to describe the named vector.
            The description of named vectors is not automatically generated by the LLM.
            Any named vectors that are not provided will not be updated.
            If None or not provided, the named vectors will not be updated.
        summary (str): The summary to update.
            The summary is a short description of the collection, generated by the LLM.
            This will replace the existing summary of the collection.
            If None or not provided, the summary will not be updated.
        mappings (dict): The mappings to update.
            The mappings are what the frontend will use to display the collection, and the associated fields.
            I.e., which fields correspond to which output fields on the frontend.
            The keys of the outer level of the dictionary are the mapping names, the values are dictionaries with the mappings.
            The inner dictionary has the keys as the collection fields, and the values as the frontend fields.
            If None or not provided, the mappings will not be updated.
        fields (list[dict]): The fields to update.
            Each element in the list is a dictionary with the following fields:
            - "name": The name of the field. (This is used to identify the field to change, the name will not change).
            - "description": The description of the field to update.
            Any fields that are not provided will not be updated.
            If None or not provided, the fields will not be updated.

    Returns:
        dict: The updated preprocessed collection.
    """

    return asyncio_run(
        edit_preprocessed_collection_async(
            collection_name, client_manager, named_vectors, summary, mappings, fields
        )
    )


async def edit_preprocessed_collection_async(
    collection_name: str,
    client_manager: ClientManager | None = None,
    named_vectors: list[dict] | None = None,
    summary: str | None = None,
    mappings: dict[str, dict[str, str]] | None = None,
    fields: list[dict[str, str] | None] | None = None,
) -> dict:
    """
    Async version of `edit_preprocessed_collection`.

    Args:
        collection_name (str): The name of the collection to edit.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.
        named_vectors (list[dict]): The named vectors to update. This has fields "name", "enabled", and "description".
            The "name" is used to identify the named vector to change (the name will not change).
            Set "enabled" to True/False to enable/disable the named vector.
            Set "description" to describe the named vector.
            The description of named vectors is not automatically generated by the LLM.
            Any named vectors that are not provided will not be updated.
            If None or not provided, the named vectors will not be updated.
        summary (str): The summary to update.
            The summary is a short description of the collection, generated by the LLM.
            This will replace the existing summary of the collection.
            If None or not provided, the summary will not be updated.
        mappings (dict): The mappings to update.
            The mappings are what the frontend will use to display the collection, and the associated fields.
            I.e., which fields correspond to which output fields on the frontend.
            The keys of the outer level of the dictionary are the mapping names, the values are dictionaries with the mappings.
            The inner dictionary has the keys as the collection fields, and the values as the frontend fields.
            If None or not provided, the mappings will not be updated.
        fields (list[dict]): The fields to update.
            Each element in the list is a dictionary with the following fields:
            - "name": The name of the field. (This is used to identify the field to change, the name will not change).
            - "description": The description of the field to update.
            Any fields that are not provided will not be updated.
            If None or not provided, the fields will not be updated.

    Returns:
        dict: The updated preprocessed collection.
    """

    if client_manager is None:
        client_manager = ClientManager()
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    async with client_manager.connect_to_async_client() as client:
        metadata_name = f"ELYSIA_METADATA__"

        # check if the collection itself exists
        if not await client.collections.exists(collection_name):
            raise Exception(f"Collection {collection_name} does not exist")

        # check if the metadata collection exists
        if not await client.collections.exists(metadata_name):
            raise Exception(f"Metadata collection does not exist")

        else:
            metadata_collection = client.collections.get(metadata_name)
            metadata = await metadata_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(collection_name),
                limit=1,
            )
            uuid = metadata.objects[0].uuid
            properties: dict = metadata.objects[0].properties  # type: ignore

        # update the named vectors
        if named_vectors is not None:
            for named_vector in named_vectors:
                for property_named_vector in properties["named_vectors"]:
                    if property_named_vector["name"] == named_vector["name"]:

                        if named_vector["enabled"] is not None:
                            property_named_vector["enabled"] = named_vector["enabled"]

                        if named_vector["description"] is not None:
                            property_named_vector["description"] = named_vector[
                                "description"
                            ]

        # update the summary
        if summary is not None:
            properties["summary"] = summary

        # update the mappings
        if mappings is not None:
            if "table" in mappings:
                mappings["table"] = {
                    field["name"]: field["name"] for field in properties["fields"]
                }

            # format all mappings
            for mapping_type, mapping in mappings.items():
                if mapping_type == "table":
                    continue

                # check if the mapping_type is valid
                if mapping_type not in rt.types_dict:
                    raise ValueError(
                        f"Invalid mapping type: {mapping_type}. Valid mapping types are: {list(rt.types_dict.keys())}"
                    )

                # check if the mapping is valid
                for field_name, field_value in mapping.items():
                    if field_name not in rt.types_dict[mapping_type]:
                        raise ValueError(
                            f"Invalid field name: {field_name} for mapping type: {mapping_type}. "
                            f"Valid fields are: {list(rt.types_dict[mapping_type].keys())}"
                        )

                # add empty fields
                for true_field in rt.types_dict[mapping_type]:
                    if true_field not in mapping:
                        mapping[true_field] = ""

            properties["mappings"] = mappings

            # check if the `conversation_id` field is in the mapping (required for conversation type)
            if "conversation" in mappings and (
                mappings["conversation"]["conversation_id"] is None
                or mappings["conversation"]["conversation_id"] == ""
            ):
                raise ValueError(
                    "Conversation type requires a conversation_id field, but none was found in the mappings for conversation. "
                )

            if "conversation" in mappings and (
                mappings["conversation"]["message_id"] is None
                or mappings["conversation"]["message_id"] == ""
            ):
                raise ValueError(
                    "Conversation type requires a message_id field, but none was found in the mappings for conversation. "
                )

            # check if there is a message type as well as conversation
            if "message" not in mappings and "conversation" in mappings:
                raise ValueError(
                    "Conversation type requires message type to also be set as a fallback."
                )

        # update the fields
        if fields is not None:
            for field in fields:
                for property_field in properties["fields"]:
                    if field is not None and property_field["name"] == field["name"]:
                        property_field["description"] = field["description"]

        format_dict_to_serialisable(properties)

        # update the collection
        await metadata_collection.data.update(uuid=uuid, properties=properties)

    if close_clients_after_completion:
        await client_manager.close_clients()

    return properties


def view_preprocessed_collection(
    collection_name: str, client_manager: ClientManager | None = None
) -> dict:
    """
    View a preprocessed collection.
    This function allows you to view the preprocessed collection generated by the preprocess function.
    It does so by querying the ELYSIA_METADATA__ collection.

    Args:
        collection_name (str): The name of the collection to view.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.

    Returns:
        dict: The preprocessed collection.
    """
    return asyncio_run(
        view_preprocessed_collection_async(collection_name, client_manager)
    )


async def view_preprocessed_collection_async(
    collection_name: str, client_manager: ClientManager | None = None
) -> dict:
    """
    Async version of `view_preprocessed_collection`.

    Args:
        collection_name (str): The name of the collection to view.
        client_manager (ClientManager): The client manager to use.
            If not provided, a new ClientManager will be created using the environment variables/configured settings.

    Returns:
        dict: The preprocessed collection.
    """

    if client_manager is None:
        client_manager = ClientManager()
        close_clients_after_completion = True
    else:
        close_clients_after_completion = False

    async with client_manager.connect_to_async_client() as client:
        metadata_name = f"ELYSIA_METADATA__"

        # check if the collection itself exists
        if not await client.collections.exists(collection_name):
            raise Exception(f"Collection {collection_name} does not exist")

        # check if the metadata collection exists
        if not await client.collections.exists(metadata_name):
            raise Exception(f"Metadata collection does not exist")

        else:
            metadata_collection = client.collections.get(metadata_name)
            metadata = await metadata_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(collection_name),
                limit=1,
            )
            if len(metadata.objects) == 0:
                raise Exception(f"Metadata for {collection_name} does not exist")

            properties: dict = metadata.objects[0].properties  # type: ignore

    if close_clients_after_completion:
        await client_manager.close_clients()

    return properties
