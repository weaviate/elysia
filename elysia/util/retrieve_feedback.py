from elysia.util.client import ClientManager
import json
import dspy
import random
from weaviate.classes.query import Filter, MetadataQuery


async def retrieve_feedback(
    client_manager: ClientManager, user_prompt: str, model: str, n: int = 6
):
    """
    Retrieve similar examples from the database.
    """

    # semantic search for similar examples
    async with client_manager.connect_to_async_client() as client:
        if not await client.collections.exists("ELYSIA_FEEDBACK__"):
            return [], []

        feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")

        # find superpositive examples
        superpositive_feedback = await feedback_collection.query.near_text(
            query=user_prompt,
            filters=Filter.all_of(
                [
                    Filter.by_property("modules_used").contains_any([model]),
                    Filter.by_property("feedback").equal(2.0),
                ]
            ),
            certainty=0.7,
            limit=n,
            return_metadata=MetadataQuery(distance=True, certainty=True),
        )

        if len(superpositive_feedback.objects) < n:

            # find positive examples
            positive_feedback = await feedback_collection.query.near_text(
                query=user_prompt,
                filters=Filter.all_of(
                    [
                        Filter.by_property("modules_used").contains_any([model]),
                        Filter.by_property("feedback").equal(1.0),
                    ],
                ),
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
