from datetime import datetime
import json

from weaviate.collections.classes.aggregate import (
    AggregateGroupByReturn,
)
from weaviate.classes.aggregate import GroupByAggregate
from weaviate.classes.query import Filter, Metrics
from weaviate.util import generate_uuid5

from elysia.tree.tree import Tree
from elysia.util.parsing import format_datetime
from elysia.api.core.log import logger
import weaviate.classes.config as wc


async def create_feedback_collection(client):
    await client.collections.create(
        "ELYSIA_FEEDBACK__",
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
                data_type=wc.DataType.OBJECT_ARRAY,
                nested_properties=[
                    wc.Property(
                        name="collection_name",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="action_name",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="code",
                        data_type=wc.DataType.OBJECT,
                        nested_properties=[
                            wc.Property(
                                name="title",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="language",
                                data_type=wc.DataType.TEXT,
                            ),
                            wc.Property(
                                name="text",
                                data_type=wc.DataType.TEXT,
                            ),
                        ],
                    ),
                    wc.Property(
                        name="return_type",
                        data_type=wc.DataType.TEXT,
                    ),
                    wc.Property(
                        name="output_type",
                        data_type=wc.DataType.TEXT,
                    ),
                ],
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
        vectorizer_config=[
            wc.Configure.NamedVectors.text2vec_openai(
                name="user_prompt",
                model="text-embedding-3-small",
                source_properties=["user_prompt"],
                vector_index_config=wc.Configure.VectorIndex.hnsw(
                    quantizer=wc.Configure.VectorIndex.Quantizer.sq(),
                ),
            ),
        ],
    )
    logger.info("Feedback collection (ELYSIA_FEEDBACK__) created!")


async def create_feedback(
    user_id: str, conversation_id: str, query_id: str, feedback: int, tree: Tree, client
):
    if not await client.collections.exists("ELYSIA_FEEDBACK__"):
        await create_feedback_collection(client)

    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")

    history = tree.history[query_id]

    # new_tasks_completed = []
    # for task_prompt in history["tree_data"].tasks_completed:
    #     new_tasks_completed.append({})
    #     new_tasks_completed[-1]["prompt"] = task_prompt["prompt"]
    #     new_tasks_completed[-1]["task"] = [
    #         {
    #             "task": h["task"],
    #             "reasoning": h["reasoning"],
    #             "todo": h["todo"],
    #             "count": float(h["count"])
    #         }
    #     for h in task_prompt["tasks"]
    #     ]

    # ensure "action" is a bool in tasks_completed
    for task_prompt in history["tree_data"].tasks_completed:
        for task in task_prompt["task"]:
            task["action"] = bool(task["action"])

    date_now = datetime.now()
    # date_now = date_now - timedelta(days=1)

    properties = {
        "user_id": user_id,
        "conversation_id": conversation_id,
        "query_id": query_id,
        "feedback": int(feedback),
        "modules_used": list(
            set([h["module_name"] for h in history["training_updates"]])
        ),
        "user_prompt": history["tree_data"].user_prompt,
        "conversation_history": history["tree_data"].conversation_history,
        "tasks_completed": history["tree_data"].tasks_completed,
        "route": history["decision_history"],
        "action_information": history["action_information"],
        "time_taken_seconds": history["time_taken_seconds"],
        "decision_time": tree.tracker.get_average_time("decision_node"),
        "base_lm_used": tree.base_lm.model,
        "complex_lm_used": tree.complex_lm.model,
        "feedback_datetime": format_datetime(date_now),
        "feedback_date": format_datetime(
            date_now.replace(hour=0, minute=0, second=0, microsecond=0)
        ),
        "training_updates": json.dumps(history["training_updates"]),
        "initialisation": history["initialisation"],
    }

    # uuid is generated based on the user_id, conversation_id, query_id ONLY
    # so if the user re-selects the same feedback, it will be updated instead of added
    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )

    try:
        if await feedback_collection.data.exists(session_uuid):
            await feedback_collection.data.update(
                properties=properties, uuid=session_uuid
            )
        else:
            await feedback_collection.data.insert(
                properties=properties, uuid=session_uuid
            )
    except Exception as e:
        logger.exception(f"Whilst inserting feedback to the feedback collection")


async def view_feedback(user_id: str, conversation_id: str, query_id: str, client):
    if not await client.collections.exists("ELYSIA_FEEDBACK__"):
        raise Exception("No feedback collection found")

    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")
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


async def remove_feedback(user_id: str, conversation_id: str, query_id: str, client):
    if not await client.collections.exists("ELYSIA_FEEDBACK__"):
        raise Exception("No feedback collection found")

    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")
    session_uuid = generate_uuid5(
        {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
        }
    )
    await feedback_collection.data.delete_by_id(uuid=session_uuid)


async def feedback_metadata(client, user_id: str):

    if not await client.collections.exists("ELYSIA_FEEDBACK__"):
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

    feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")

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

        filters = Filter.by_property("feedback").equal(feedback_value)

        # by value
        agg_feedback_count_i = await feedback_collection.aggregate.over_all(
            filters=filters, return_metrics=[Metrics("feedback").integer(count=True)]
        )
        feedback_by_value[feedback_name] = agg_feedback_count_i.properties[
            "feedback"
        ].count

        # by date
        agg_feedback_i = await feedback_collection.aggregate.over_all(
            group_by=GroupByAggregate(prop="feedback_date"),
            filters=filters,
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


# if __name__ == "__main__":
#     import os
#     import weaviate
#     from weaviate.classes.init import Auth
#     client = weaviate.connect_to_weaviate_cloud(
#         cluster_url = os.environ.get("WCD_URL"),
#         auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
#         headers={"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
#     )
#     print(feedback_metadata(client))
