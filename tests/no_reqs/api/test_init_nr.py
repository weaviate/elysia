import pytest
from fastapi.responses import JSONResponse
import os
import json

from elysia.api.core.log import set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.init import initialise_tree, initialise_user
from elysia.api.api_types import InitialiseTreeData


def read_response(response: JSONResponse):
    return json.loads(response.body)


class TestTree:

    @pytest.mark.asyncio
    async def test_initialise_user(self):
        user_manager = get_user_manager()
        user_id = "test_new_user"

        # Initialise when a user does not exist
        out = await initialise_user(
            user_id,
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""
        assert response["user_exists"] is False

        # check config options are environment variables
        if "BASE_MODEL" in os.environ:
            assert response["config"]["settings"]["BASE_MODEL"] == os.getenv(
                "BASE_MODEL"
            )
        if "OPENAI_API_KEY" in os.environ:
            assert response["config"]["settings"]["API_KEYS"][
                "openai_api_key"
            ] == os.getenv("OPENAI_API_KEY")

        # Initialise when a user exists
        out = await initialise_user(
            user_id,
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""
        assert response["user_exists"] is True

    @pytest.mark.asyncio
    async def test_tree(self):

        user_manager = get_user_manager()
        user_id = "test_user"
        conversation_id = "test_conversation"

        out = await initialise_user(
            user_id,
            user_manager,
        )
        response = read_response(out)
        assert response["error"] == ""

        out = await initialise_tree(
            user_id,
            conversation_id,
            InitialiseTreeData(
                low_memory=True,
            ),
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""

        out = await initialise_tree(
            user_id + "2",
            conversation_id + "2",
            InitialiseTreeData(
                low_memory=True,
            ),
            user_manager,
        )

        response = read_response(out)
        assert response["error"] != ""  # should error as user doesn't exist
