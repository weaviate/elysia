from datetime import datetime
import json
import dspy
import random

from weaviate.collections.classes.aggregate import (
    AggregateGroupByReturn,
)
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Filter, Metrics, MetadataQuery
from weaviate.classes.config import Configure
from weaviate.client import WeaviateAsyncClient
from weaviate.util import generate_uuid5
import weaviate.classes.config as wc

from elysia.util.client import ClientManager
from elysia.api.core.log import logger


async def create_feedback_collection(
    client: WeaviateAsyncClient, collection_name="ELYSIA_FEEDBACK__"
):
    await client.collections.create(
        collection_name,
        properties=[
            # session data
            wc.Property(
                name="user_id",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="conversation_id",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="query_id",
                data_type=wc.DataType.TEXT,
            ),
            # feedback value (between -2, -1, 1, 2)
            wc.Property(
                name="feedback",
                data_type=wc.DataType.NUMBER,
            ),
            # track which models were used
            wc.Property(
                name="modules_used",
                data_type=wc.DataType.TEXT_ARRAY,
            ),
            # Tree data (except available_information)
            wc.Property(
                name="user_prompt",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="conversation_history",
                data_type=wc.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wc.Property(
                        name="role",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="content",
                        data_type=wc.DataType.TEXT,
                    ),
                ],
            ),
            wc.Property(
                name="tasks_completed",
                data_type=wc.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wc.Property(
                        name="prompt",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="tasks",
                        data_type=wc.DataType.OBJECT_ARRAY,
                        nested_properties=[
                            wc.Property(
                                name="task",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="reasoning",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="todo",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(name="action", data_type=wc.DataType.BOOL),
                            wc.Property(
                                name="count",
                                data_type=wc.DataType.NUMBER,
                            ),
                            wc.Property(
                                name="extra_string",
                                data_type=wc.DataType.TEXT,
                            ),
                        ],
                    ),
                ],
            ),
            # Extra specifics
            wc.Property(
                name="route",
                data_type=wc.DataType.TEXT_ARRAY,
            ),
            wc.Property(
                name="action_information",
                data_type=wc.DataType.TEXT,
            ),
            # metadata
            wc.Property(
                name="time_taken_seconds",
                data_type=wc.DataType.NUMBER,
            ),
            wc.Property(
                name="base_lm_used",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="complex_lm_used",
                data_type=wc.DataType.TEXT,
            ),
            wc.Property(
                name="feedback_date",
                data_type=wc.DataType.DATE,
            ),
            wc.Property(
                name="decision_time",
                data_type=wc.DataType.NUMBER,
            ),
            # dump training_updates as string
            wc.Property(name="training_updates", data_type=wc.DataType.TEXT),
        ],
        vector_config=[
            wc.Configure.Vectors.text2vec_openai(
                name="user_prompt",
                model="text-embedding-3-small",
                source_properties=["user_prompt"],
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    quantizer=wc.Configure.VectorIndex.Quantizer.sq(),
                ),
            ),
        ],
        multi_tenancy_config=Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=True,
            auto_tenant_activation=True,
        ),
    )
    logger.info(f"Feedback collection (ELYSIA_FEEDBACK__) created!")


async def view_feedback(
    user_id: str,
    conversation_id: str,
    query_id: str,
    client: WeaviateAsyncClient,
    collection_name="ELYSIA_FEEDBACK__",
):
    if not await client.collections.exists(collection_name):
        raise Exception("No feedback collection found")

    base_feedback_collection = client.collections.get(collection_name)
    if not await base_feedback_collection.tenants.exists(user_id):
        raise Exception("User ID not in feedback collection")

    feedback_collection = base_feedback_collection.with_tenant(user_id)

    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )

    feedback_object = await feedback_collection.query.fetch_object_by_id(
        uuid=session_uuid
    )

    return feedback_object.properties


async def remove_feedback(
    user_id: str,
    conversation_id: str,
    query_id: str,
    client: WeaviateAsyncClient,
    collection_name="ELYSIA_FEEDBACK__",
):
    if not await client.collections.exists(collection_name):
        raise Exception("No feedback collection found")

    base_feedback_collection = client.collections.get(collection_name)
    if not await base_feedback_collection.tenants.exists(user_id):
        raise Exception("User ID not in feedback collection")

    feedback_collection = base_feedback_collection.with_tenant(user_id)

    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )
    await feedback_collection.data.delete_by_id(uuid=session_uuid)


async def feedback_metadata(
    client: WeaviateAsyncClient,
    user_id: str,
    collection_name="ELYSIA_FEEDBACK__",
):

    if not await client.collections.exists(collection_name):
        return {
            "error": "",
            "total_feedback": 0,
            "feedback_by_value": {
                "negative": 0,
                "positive": 0,
                "superpositive": 0,
            },
            "feedback_by_date": {},
        }

    base_feedback_collection = client.collections.get(collection_name)

    if not await base_feedback_collection.tenants.exists(user_id):
        return {
            "error": "",
            "total_feedback": 0,
            "feedback_by_value": {
                "negative": 0,
                "positive": 0,
                "superpositive": 0,
            },
            "feedback_by_date": {},
        }

    feedback_collection = base_feedback_collection.with_tenant(user_id)

    all_aggregate = await feedback_collection.aggregate.over_all(total_count=True)
    total_feedback = all_aggregate.total_count

    feedback_values = {
        "negative": 0.0,
        "positive": 1.0,
        "superpositive": 2.0,
    }

    feedback_by_date = {}
    feedback_by_value = {}
    for feedback_name, feedback_value in feedback_values.items():

        # by value
        agg_feedback_count_i = await feedback_collection.aggregate.over_all(
            filters=Filter.by_property("feedback").equal(feedback_value),
            return_metrics=[Metrics("feedback").integer(count=True)],
        )
        feedback_by_value[feedback_name] = agg_feedback_count_i.properties[
            "feedback"
        ].count

        # by date
        agg_feedback_i = await feedback_collection.aggregate.over_all(
            group_by=GroupByAggregate(prop="feedback_date"),
            filters=Filter.by_property("feedback").equal(feedback_value),
            return_metrics=[Metrics("feedback").number(count=True)],
        )

        # if there is a property
        if isinstance(agg_feedback_i, AggregateGroupByReturn):

            for date_group in agg_feedback_i.groups:

                date_val = datetime.fromisoformat(date_group.grouped_by.value).strftime(  # type: ignore
                    "%Y-%m-%d"
                )

                if date_val not in feedback_by_date:
                    feedback_by_date[date_val] = {
                        "negative": 0,
                        "positive": 0,
                        "superpositive": 0,
                    }

                feedback_by_date[date_val][feedback_name] += date_group.properties[
                    "feedback"
                ].count  # type: ignore

    # fill in empties with zeros
    for date in feedback_by_date:
        for feedback_name in feedback_values:
            if feedback_name not in feedback_by_date[date]:
                feedback_by_date[date][feedback_name] = 0

    agg_feedback = await feedback_collection.aggregate.over_all(
        group_by=GroupByAggregate(prop="feedback_date"),
        return_metrics=[Metrics("feedback").number(mean=True, count=True)],
    )

    if isinstance(agg_feedback, AggregateGroupByReturn):

        for date_group in agg_feedback.groups:

            date_val = datetime.fromisoformat(date_group.grouped_by.value).strftime(  # type: ignore
                "%Y-%m-%d"
            )

            feedback_by_date[date_val]["mean"] = date_group.properties["feedback"].mean  # type: ignore
            feedback_by_date[date_val]["count"] = date_group.properties[
                "feedback"
            ].count  # type: ignore

    return {
        "total_feedback": total_feedback,
        "feedback_by_value": feedback_by_value,
        "feedback_by_date": feedback_by_date,
    }


async def retrieve_feedback(
    client_manager: ClientManager,
    user_prompt: str,
    model: str,
    user_id: str,
    n: int = 6,
    collection_name: str = "ELYSIA_FEEDBACK__",
):
    """
    Retrieve similar examples from the database.
    """

    # semantic search for similar examples
    async with client_manager.connect_to_async_client() as client:

        if not await client.collections.exists(collection_name):
            return [], []

        base_feedback_collection = client.collections.get(collection_name)

        if not await base_feedback_collection.tenants.exists(user_id):
            return [], []

        feedback_collection = base_feedback_collection.with_tenant(user_id)

        filters = Filter.all_of(
            [
                Filter.by_property("modules_used").contains_any([model]),
                Filter.by_property("feedback").equal(2.0),
            ]
        )

        # find superpositive examples
        superpositive_feedback = await feedback_collection.query.near_text(
            query=user_prompt,
            filters=filters,
            certainty=0.7,
            limit=n,
            return_metadata=MetadataQuery(distance=True, certainty=True),
        )

        if len(superpositive_feedback.objects) < n:

            filters = Filter.all_of(
                [
                    Filter.by_property("modules_used").contains_any([model]),
                    Filter.by_property("feedback").equal(1.0),
                ]
            )

            # find positive examples
            positive_feedback = await feedback_collection.query.near_text(
                query=user_prompt,
                filters=filters,
                certainty=0.7,
                limit=n,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )

            feedback_objects = superpositive_feedback.objects
            feedback_objects.extend(
                positive_feedback.objects[: (n - len(superpositive_feedback.objects))]
            )

        else:
            feedback_objects = superpositive_feedback.objects

    # get training updates
    training_updates = [
        json.loads(f.properties["training_updates"])  # type: ignore
        for f in feedback_objects
    ]
    uuids = [str(f.uuid) for f in feedback_objects]

    relevant_updates = []
    relevant_uuids = []
    for i, update in enumerate(training_updates):
        for inner_update in update:
            if inner_update["module_name"] == model:
                relevant_updates.append(inner_update)
        relevant_uuids.append(uuids[i])

    # take max n randomly selected updates
    random.shuffle(relevant_updates)
    relevant_updates = relevant_updates[:n]

    examples = []
    for update in relevant_updates:
        examples.append(
            dspy.Example(
                {
                    **{k: v for k, v in update["inputs"].items()},
                    **{k: v for k, v in update["outputs"].items()},
                }
            ).with_inputs(
                *update["inputs"].keys(),
            )
        )

    return examples, relevant_uuids
