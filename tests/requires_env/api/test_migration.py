from weaviate.client import WeaviateAsyncClient
from datetime import datetime
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import Filter
from weaviate.client import WeaviateAsyncClient
from weaviate.classes.aggregate import GroupByAggregate

from elysia.util.parsing import format_datetime
from elysia.util.client import ClientManager


from elysia.api.routes.utils import migrate
from elysia.api.api_types import MigrateDataData, SaveConfigUserData
from elysia.api.routes.init import initialise_user
from elysia.api.routes.user_config import save_config_user
from elysia.api.services.user import UserManager
import os
from fastapi.responses import JSONResponse
import json


def read_response(response: JSONResponse):
    return json.loads(response.body)


async def manual_test_migration():
    user_manager = UserManager()

    # initialise a user
    user_id = "test_user_migration"
    response = await initialise_user(user_id, user_manager)
    response = read_response(response)

    # set the user's config to the testing URL
    response = await save_config_user(
        user_id,
        "test_config_id",
        SaveConfigUserData(
            name="test_name",
            frontend_config={
                "save_location_wcd_url": os.getenv("TESTING_WCD_URL"),
                "save_location_wcd_api_key": os.getenv("TESTING_WCD_API_KEY"),
            },
            default=True,
            config={"logging_level": "DEBUG"},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)

    # this cluster uses old collections, so not supported
    assert not response["elysia_collections_supported"]

    # initialise the same user and check
    response = await initialise_user(user_id, user_manager)
    response = read_response(response)
    assert not response["correct_settings"]["elysia_collections_supported"]

    # reset user manager and check
    user_manager = UserManager()
    response = await initialise_user(user_id, user_manager)
    response = read_response(response)
    assert not response["correct_settings"]["elysia_collections_supported"]

    # migrate collections
    response = await migrate(user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == ""

    # initialise the same user and check
    response = await initialise_user(user_id, user_manager)
    response = read_response(response)
    assert response["correct_settings"]["elysia_collections_supported"]


if __name__ == "__main__":
    import asyncio

    asyncio.run(manual_test_migration())
