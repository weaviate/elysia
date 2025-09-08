import pytest

from datetime import datetime

from elysia.api.routes.query import process
from elysia.api.dependencies.common import get_user_manager
from elysia.util.client import ClientManager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.feedback import (
    run_add_feedback,
    run_feedback_metadata,
    run_remove_feedback,
)

from elysia.api.api_types import (
    InitialiseTreeData,
    QueryData,
    AddFeedbackData,
    RemoveFeedbackData,
)


from fastapi.responses import JSONResponse
import json
from weaviate.util import generate_uuid5


def read_response(response: JSONResponse):
    return json.loads(response.body)


from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


async def initialise_user_and_tree(user_id: str, conversation_id: str):
    user_manager = get_user_manager()

    response = await initialise_user(
        user_id,
        user_manager,
    )

    response = await initialise_tree(
        user_id,
        conversation_id,
        InitialiseTreeData(
            low_memory=False,
        ),
        user_manager,
    )


@pytest.mark.asyncio
async def test_full_feedback_cycle():
    """
    Test the full feedback cycle.
    Add feedback, check metadata to see if it was added, remove feedback, check metadata to see if it was removed.
    """

    user_id = "test_user_full_feedback_cycle"
    conversation_id = "test_conversation_full_feedback_cycle"
    query_id = "test_query_full_feedback_cycle"
    user_manager = get_user_manager()
    websocket = fake_websocket()

    await initialise_user_and_tree(user_id, conversation_id)

    out = await process(
        QueryData(
            user_id=user_id,
            conversation_id=conversation_id,
            query="hi!",
            query_id=query_id,
            collection_names=[],
        ).model_dump(),
        websocket,
        user_manager,
    )

    response = await run_feedback_metadata(
        user_id=user_id,
        user_manager=user_manager,
    )

    assert read_response(response)["error"] == ""

    # record number of feedbacks
    num_feedbacks = read_response(response)["total_feedback"]

    # number of positives
    num_positives = read_response(response)["feedback_by_value"]["positive"]

    # number of dates
    date = datetime.now().strftime("%Y-%m-%d")
    if date in read_response(response)["feedback_by_date"]:
        num_feedbacks_on_this_date = read_response(response)["feedback_by_date"][date][
            "count"
        ]
    else:
        num_feedbacks_on_this_date = 0

    # check if the feedback exists already
    async with user_manager.users[user_id][
        "client_manager"
    ].connect_to_async_client() as client:
        feedback_collection = client.collections.get("ELYSIA_FEEDBACK__")
        session_uuid = generate_uuid5(
            {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "query_id": query_id,
            }
        )
        feedback_exists = await feedback_collection.data.exists(uuid=session_uuid)

    response = await run_add_feedback(
        AddFeedbackData(
            user_id=user_id,
            conversation_id=conversation_id,
            query_id=query_id,
            feedback=1,
        ),
        user_manager,
    )

    assert read_response(response)["error"] == ""

    response = await run_feedback_metadata(
        user_id=user_id,
        user_manager=user_manager,
    )

    assert read_response(response)["error"] == ""

    # number of positives should have increased by 1
    if not feedback_exists:
        assert (
            read_response(response)["feedback_by_value"]["positive"]
            == num_positives + 1
        )

        # number of feedbacks on this date should have increased by 1
        assert (
            read_response(response)["feedback_by_date"][date]["count"]
            == num_feedbacks_on_this_date + 1
        )

        # number of feedbacks should have increased by 1
        assert read_response(response)["total_feedback"] == num_feedbacks + 1
    else:
        assert read_response(response)["feedback_by_value"]["positive"] == num_positives
        assert (
            read_response(response)["feedback_by_date"][date]["count"]
            == num_feedbacks_on_this_date
        )
        assert read_response(response)["total_feedback"] == num_feedbacks

    response = await run_remove_feedback(
        RemoveFeedbackData(
            user_id=user_id,
            conversation_id=conversation_id,
            query_id=query_id,
        ),
        user_manager,
    )

    assert read_response(response)["error"] == ""

    response = await run_feedback_metadata(
        user_id=user_id,
        user_manager=user_manager,
    )

    assert read_response(response)["error"] == ""

    if not feedback_exists:
        # number of positives should have decreased by 1
        assert read_response(response)["feedback_by_value"]["positive"] == num_positives

        # number of feedbacks on this date should have decreased by 1 or been removed
        if date in read_response(response)["feedback_by_date"]:
            assert (
                read_response(response)["feedback_by_date"][date]["count"]
                == num_feedbacks_on_this_date
            )

        # number of feedbacks should have decreased by 1
        assert read_response(response)["total_feedback"] == num_feedbacks
    else:
        assert (
            read_response(response)["feedback_by_value"]["positive"]
            == num_positives - 1
        )
        assert read_response(response)["total_feedback"] == num_feedbacks - 1

        # number of feedbacks on this date should have decreased by 1 or been removed
        if date in read_response(response)["feedback_by_date"]:
            assert (
                read_response(response)["feedback_by_date"][date]["count"]
                == num_feedbacks_on_this_date - 1
            )
