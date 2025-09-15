import os
import pytest

from elysia.config import Settings
from elysia.api.routes.init import initialise_user
from elysia.api.routes.user_config import (
    load_a_config,
    save_config_user,
    load_config_user,
    new_user_config,
    get_current_user_config,
)
from elysia.api.api_types import SaveConfigUserData
from fastapi.responses import JSONResponse
import json
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from weaviate.util import generate_uuid5


def read_response(response: JSONResponse):
    return json.loads(response.body)


async def delete_config_after_completion(
    user_manager: UserManager, user_id: str, config_id: str
):
    client_manager = user_manager.users[user_id]["client_manager"]
    async with client_manager.connect_to_async_client() as client:
        if not await client.collections.exists("ELYSIA_CONFIG__"):
            return

        collection = client.collections.get("ELYSIA_CONFIG__")
        uuid = generate_uuid5(config_id)
        if await collection.data.exists(uuid):
            await collection.data.delete_by_id(uuid)

    if os.path.exists(f"elysia/api/user_configs/frontend_config_{user_id}.json"):
        os.remove(f"elysia/api/user_configs/frontend_config_{user_id}.json")


@pytest.mark.asyncio
async def test_encryption():

    user_id = "test_encryption"

    user_manager = get_user_manager()

    # initialise the user
    response = await initialise_user(user_id=user_id, user_manager=user_manager)
    response = read_response(response)

    # create a new config
    response = await new_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    config_id = response["config"]["id"]

    # check the WCD_API_KEY
    assert response["config"]["settings"]["WCD_API_KEY"] == os.getenv("WCD_API_KEY")

    # change config, don't change anything
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=response["config"]["name"],
            default=True,
            config=response["config"],
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["config"]["settings"]["WCD_API_KEY"] == os.getenv("WCD_API_KEY")

    # get the config
    response = await load_a_config(
        user_id=user_id, config_id=config_id, user_manager=user_manager
    )
    response = read_response(response)
    assert response["config"]["settings"]["WCD_API_KEY"] == os.getenv("WCD_API_KEY")


@pytest.mark.asyncio
async def test_incorrect_api_keys():

    user_id = "test_incorrect_api_keys"
    user_manager = get_user_manager()

    correct_api_key = os.getenv("OPENROUTER_API_KEY")
    incorrect_api_key = "incorrect"

    # initialise the user
    response = await initialise_user(user_id=user_id, user_manager=user_manager)
    response = read_response(response)

    # create a new config
    response = await new_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    config_id = response["config"]["id"]

    # change config, change the OPENROUTER_API_KEY
    new_config = response["config"]
    # new_config["settings"]["OPENROUTER_API_KEY"] = incorrect_api_key
    new_config["settings"]["API_KEYS"]["openrouter_api_key"] = incorrect_api_key

    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=response["config"]["name"],
            default=True,
            config=new_config,
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert (
        response["config"]["settings"]["API_KEYS"]["openrouter_api_key"]
        == incorrect_api_key
    )

    # load a config
    response = await get_current_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert (
        response["config"]["settings"]["API_KEYS"]["openrouter_api_key"]
        == incorrect_api_key
    )


@pytest.mark.asyncio
async def test_change_models():

    user_id = "test_change_models"
    user_manager = get_user_manager()

    # initialise the user
    response = await initialise_user(user_id=user_id, user_manager=user_manager)
    response = read_response(response)

    # create a new config
    response = await new_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    config_id = response["config"]["id"]

    # change config, change the models
    new_config = response["config"]
    new_config["settings"]["BASE_MODEL"] = "gpt-4o"
    new_config["settings"]["COMPLEX_MODEL"] = "gpt-4o"
    new_config["settings"]["BASE_PROVIDER"] = "openrouter/openai"
    new_config["settings"]["COMPLEX_PROVIDER"] = "openrouter/openai"

    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=response["config"]["name"],
            default=True,
            config=new_config,
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["config"]["settings"]["BASE_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["COMPLEX_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["BASE_PROVIDER"] == "openrouter/openai"
    assert response["config"]["settings"]["COMPLEX_PROVIDER"] == "openrouter/openai"

    # load a config
    response = await get_current_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["config"]["settings"]["BASE_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["COMPLEX_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["BASE_PROVIDER"] == "openrouter/openai"
    assert response["config"]["settings"]["COMPLEX_PROVIDER"] == "openrouter/openai"
