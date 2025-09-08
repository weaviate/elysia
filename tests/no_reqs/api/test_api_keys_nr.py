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

from uuid import uuid4


def read_response(response: JSONResponse):
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_set_api_keys():

    user_id = f"test_{uuid4()}"
    user_manager = get_user_manager()

    openrouter_api_key = "some-random-key"

    # initialise the user
    response = await initialise_user(user_id=user_id, user_manager=user_manager)
    response = read_response(response)

    # create a new config
    response = await new_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    config_id = response["config"]["id"]

    new_config = response["config"]
    new_config["settings"]["API_KEYS"]["openrouter_api_key"] = openrouter_api_key

    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=response["config"]["name"],
            default=True,
            config=new_config,
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert (
        response["config"]["settings"]["API_KEYS"]["openrouter_api_key"]
        == openrouter_api_key
    )

    # load a config
    response = await get_current_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert (
        response["config"]["settings"]["API_KEYS"]["openrouter_api_key"]
        == openrouter_api_key
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
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert response["config"]["settings"]["BASE_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["COMPLEX_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["BASE_PROVIDER"] == "openrouter/openai"
    assert response["config"]["settings"]["COMPLEX_PROVIDER"] == "openrouter/openai"

    # load a config
    response = await get_current_user_config(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert response["config"]["settings"]["BASE_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["COMPLEX_MODEL"] == "gpt-4o"
    assert response["config"]["settings"]["BASE_PROVIDER"] == "openrouter/openai"
    assert response["config"]["settings"]["COMPLEX_PROVIDER"] == "openrouter/openai"
