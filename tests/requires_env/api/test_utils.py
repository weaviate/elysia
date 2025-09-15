import pytest
import json
from fastapi.responses import JSONResponse


from elysia.api.core.log import set_log_level
from elysia.api.dependencies.common import get_user_manager

set_log_level("CRITICAL")

from elysia.api.routes.init import initialise_tree
from elysia.api.routes.utils import (
    follow_up_suggestions,
)
from elysia.api.routes.init import initialise_user
from elysia.api.api_types import (
    FollowUpSuggestionsData,
    InitialiseTreeData,
)


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


class TestUtils:

    @pytest.mark.asyncio
    async def test_follow_up_suggestions(self):
        user_id = "test_user_follow_up_suggestions"
        conversation_id = "test_conversation_follow_up_suggestions"

        user_manager = get_user_manager()

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

        out = await follow_up_suggestions(
            FollowUpSuggestionsData(
                user_id=user_id,
                conversation_id=conversation_id,
            ),
            user_manager,
        )

        response = read_response(out)
        assert response["error"] == ""


if __name__ == "__main__":
    import asyncio

    asyncio.run(TestUtils().test_follow_up_suggestions())
