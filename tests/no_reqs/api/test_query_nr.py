import asyncio
import pytest
from fastapi.responses import JSONResponse

import json

from elysia.api.core.log import logger, set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.query import process
from elysia.api.api_types import QueryData, InitialiseTreeData, SaveConfigUserData
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import save_config_user

from uuid import uuid4


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
            low_memory=True,
        ),
        user_manager,
    )


@pytest.mark.asyncio
async def test_query():
    """
    Test the query endpoint.
    """

    user_id = f"test_{uuid4()}"
    conversation_id = f"test_{uuid4()}"
    query_id = f"test_{uuid4()}"
    config_id = f"test_{uuid4()}"

    websocket = fake_websocket()

    await initialise_user_and_tree(user_id, conversation_id)

    # set config for models
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name="test_config",
            default=True,
            config={
                "settings": {
                    "BASE_MODEL": "gpt-4o-mini",
                    "BASE_PROVIDER": "openai",
                    "COMPLEX_MODEL": "gpt-4o",
                    "COMPLEX_PROVIDER": "openai",
                },
            },
            frontend_config={
                "save_trees_to_weaviate": False,
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=get_user_manager(),
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]

    out = await process(
        QueryData(
            user_id=user_id,
            conversation_id=conversation_id,
            query="hi!",
            query_id=query_id,
            collection_names=[
                "Test_ELYSIA_collection_1",
                "Test_ELYSIA_collection_2",
            ],
        ).model_dump(),
        websocket,
        get_user_manager(),
    )

    # check all payloads are valid
    ner_found = False
    title_found = False
    complete_found = False
    for i, result in enumerate(websocket.results):
        if result["type"] == "ner":
            ner_found = True
        if result["type"] == "title":
            title_found = True
        if result["type"] == "completed":
            complete_found = True

        assert result["type"] != "error", result["payload"]["text"]
        assert isinstance(result, dict)
        assert "type" in result
        assert "id" in result
        assert "conversation_id" in result
        assert "query_id" in result
        assert "payload" in result
        assert isinstance(result["payload"], dict)
        if "objects" in result["payload"]:
            for obj in result["payload"]["objects"]:
                assert isinstance(obj, dict)
        if "metadata" in result["payload"]:
            assert isinstance(result["payload"]["metadata"], dict)
        if "text" in result["payload"]:
            assert isinstance(result["payload"]["text"], str)

        if result["type"] == "ner":
            assert isinstance(result["payload"]["text"], str)
            assert isinstance(result["payload"]["entity_spans"], list)
            assert isinstance(result["payload"]["noun_spans"], list)
            assert isinstance(result["payload"]["error"], str)
            assert result["payload"]["error"] == ""

        if result["type"] == "title":
            assert isinstance(result["payload"]["title"], str)
            assert isinstance(result["payload"]["error"], str)
            assert result["payload"]["error"] == ""

        if result["type"] == "completed":
            assert i == len(websocket.results) - 1
            assert websocket.results[i - 1]["type"] == "title"

    assert ner_found
    assert title_found
    assert complete_found
