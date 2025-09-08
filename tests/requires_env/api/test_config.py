import os
from fastapi.responses import JSONResponse
from pathlib import Path
from logging import Logger
import json
import random
import pytest
from weaviate.util import generate_uuid5
from elysia.api.api_types import (
    InitialiseTreeData,
    SaveConfigUserData,
    SaveConfigTreeData,
    UpdateFrontendConfigData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_tree, initialise_user
from elysia.api.routes.user_config import (
    load_config_user,
    delete_config,
    save_config_user,
    new_user_config,
    list_configs,
    update_frontend_config,
    get_models,
)
from elysia.api.routes.tree_config import (
    load_config_tree,
    change_config_tree,
    new_tree_config,
)
from elysia.tree.tree import Tree
from elysia.api.core.log import logger, set_log_level
from elysia.api.services.tree import TreeManager
from elysia.api.services.user import UserManager
from elysia.api.utils.config import FrontendConfig

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


async def delete_config_after_completion(
    user_manager: UserManager, user_id: str, config_id: str
):
    client_manager = user_manager.users[user_id][
        "frontend_config"
    ].save_location_client_manager

    if client_manager.is_client:
        async with client_manager.connect_to_async_client() as client:
            if not await client.collections.exists("ELYSIA_CONFIG__"):
                return

            collection = client.collections.get("ELYSIA_CONFIG__")
            uuid = generate_uuid5(config_id)
            if await collection.data.exists(uuid):
                await collection.data.delete_by_id(uuid)

    if os.path.exists(f"elysia/api/user_configs/frontend_config_{user_id}.json"):
        os.remove(f"elysia/api/user_configs/frontend_config_{user_id}.json")


async def initialise_user_and_tree(user_id: str, conversation_id: str):
    user_manager = get_user_manager()

    response = await initialise_user(
        user_id,
        user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    response = await initialise_tree(
        user_id,
        conversation_id,
        InitialiseTreeData(
            low_memory=False,
        ),
        user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""


@pytest.mark.asyncio
async def test_set_save_location():
    """
    Setting the WCD_URL/WCD_API_KEY in the frontend config should be saved to the user config.
    """
    user_manager = get_user_manager()
    user_id = "test_user_set_save_location"
    conversation_id = "test_conversation_set_save_location"
    config_id = "test_config_set_save_location"
    config_name = "Test set save location"

    await initialise_user_and_tree(user_id, conversation_id)

    # the user should have a save location from the .env
    user = await user_manager.get_user_local(user_id)
    assert user["frontend_config"].save_location_wcd_url == os.getenv("WCD_URL")
    assert user["frontend_config"].save_location_wcd_api_key == os.getenv("WCD_API_KEY")

    # update the save location to an invalid one
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={},
            frontend_config={
                "save_location_wcd_url": "test_incorrect_wcd_url",
                "save_location_wcd_api_key": "test_incorrect_wcd_api_key",
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)

    # but the user should have a new save location
    user = await user_manager.get_user_local(user_id)
    assert user["frontend_config"].save_location_wcd_url == "test_incorrect_wcd_url"
    assert (
        user["frontend_config"].save_location_wcd_api_key
        == "test_incorrect_wcd_api_key"
    )
    warning_found = False
    for warning in response["warnings"]:
        warning_found = "Could not connect to Weaviate" in warning
    assert warning_found

    # same if no api keys are provided
    # update the save location to an invalid one
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={},
            frontend_config={
                "save_location_wcd_url": "",
                "save_location_wcd_api_key": "",
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)

    warning_found = False
    for warning in response["warnings"]:
        warning_found = "No valid destination for config save location found" in warning
    assert warning_found


@pytest.mark.asyncio
async def test_smart_setup_models_user():
    """
    Test that models are set based on supplied api keys.
    """
    user_manager = get_user_manager()
    user_id = "test_user_smart_setup_models_user"
    conversation_id = "test_conversation_smart_setup_models_user"

    await initialise_user_and_tree(user_id, conversation_id)

    # check that the tree manager has the correct settings
    tree_manager = user_manager.users[user_id]["tree_manager"]

    assert tree_manager.settings.BASE_MODEL is not None
    assert tree_manager.settings.COMPLEX_MODEL is not None
    assert tree_manager.settings.BASE_PROVIDER is not None
    assert tree_manager.settings.COMPLEX_PROVIDER is not None

    # should default use environment variables
    assert tree_manager.settings.WCD_URL == os.getenv("WCD_URL")
    assert tree_manager.settings.WCD_API_KEY == os.getenv("WCD_API_KEY")
    assert tree_manager.config.style != ""
    assert tree_manager.config.agent_description != ""
    assert tree_manager.config.end_goal != ""
    assert tree_manager.config.branch_initialisation != ""


@pytest.mark.asyncio
async def test_smart_setup_tree():
    """
    Test that models are set based on api keys in the .env for a tree.
    """
    user_manager = get_user_manager()
    user_id = "test_user_smart_setup_tree"
    conversation_id = "test_conversation_smart_setup_tree"

    await initialise_user_and_tree(user_id, conversation_id)
    tree: Tree = await user_manager.get_tree(user_id, conversation_id)

    # and the tree should have the correct models
    assert tree.settings.BASE_MODEL is not None
    assert tree.settings.COMPLEX_MODEL is not None
    assert tree.settings.BASE_PROVIDER is not None
    assert tree.settings.COMPLEX_PROVIDER is not None


@pytest.mark.asyncio
async def test_save_config():
    """
    Test saving a user config gets added to the Weaviate database.
    """
    user_manager = get_user_manager()
    user_id = "test_user_save_config"
    conversation_id = "test_conversation_save_config"
    config_id = f"test_config_conversation_save_config"
    config_name = "Test config save"

    await initialise_user_and_tree(user_id, conversation_id)

    # create a config and save
    user = await user_manager.get_user_local(user_id)
    tree_manager: TreeManager = user["tree_manager"]

    new_settings = {
        "WCD_URL": os.getenv("TESTING_WCD_URL"),
        "WCD_API_KEY": os.getenv("TESTING_WCD_API_KEY"),
    }

    # check that the first WCD_URL etc is the env one and not the new_settings ones
    config = user["tree_manager"].config.to_json()
    assert config["settings"]["WCD_URL"] == os.getenv("WCD_URL")
    assert config["settings"]["WCD_API_KEY"] == os.getenv("WCD_API_KEY")

    assert config["settings"]["WCD_URL"] != new_settings["WCD_URL"]
    assert config["settings"]["WCD_API_KEY"] != new_settings["WCD_API_KEY"]

    # change config
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=False,
            config={
                "settings": new_settings,
                "style": "Test style",
                "agent_description": "Test description",
                "end_goal": "Test end goal",
                "branch_initialisation": tree_manager.config.branch_initialisation,
            },
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]

    # verify the config was saved by listing all configs
    response = await list_configs(
        user_id=user_id,
        user_manager=user_manager,
    )

    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert config_id in [c["config_id"] for c in response["configs"]]


@pytest.mark.asyncio
async def test_load_config_user():
    """
    Test loading a user config from the Weaviate database.
    """
    user_manager = get_user_manager()
    user_id = "test_user_load_config_user"
    conversation_id = "test_conversation_load_config_user"
    config_id = f"test_config_user"
    config_name = "Test config user"

    await initialise_user_and_tree(user_id, conversation_id)

    # create a config to save with custom settings
    settings = {
        "BASE_MODEL": "gpt-4o-mini",
        "COMPLEX_MODEL": "gpt-4o",
        "BASE_PROVIDER": "openai",
        "COMPLEX_PROVIDER": "openai",
    }
    custom_style = "Technical and precise."
    custom_agent_description = "You are a technical assistant focusing on details."
    custom_end_goal = "Provide technically accurate information to users."
    custom_branch_init = "multi_branch"

    # change user settings to these values
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": settings,
                "style": custom_style,
                "agent_description": custom_agent_description,
                "end_goal": custom_end_goal,
                "branch_initialisation": custom_branch_init,
            },
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # now change settings to something else WITHOUT saving
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=False,
            config={
                "settings": {
                    "BASE_MODEL": "claude-3-haiku",
                    "COMPLEX_MODEL": "claude-3-opus",
                    "BASE_PROVIDER": "anthropic",
                    "COMPLEX_PROVIDER": "anthropic",
                },
                "style": "Conversational and friendly.",
                "agent_description": "You are a friendly assistant.",
                "end_goal": "Provide helpful and friendly responses.",
                "branch_initialisation": "one_branch",
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Load the config back for the user
    response = await load_config_user(
        user_id=user_id,
        config_id=config_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Check the user settings were updated back to the saved config
    user = await user_manager.get_user_local(user_id)
    tree_manager = user["tree_manager"]

    assert tree_manager.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree_manager.settings.COMPLEX_MODEL == "gpt-4o"
    assert tree_manager.settings.BASE_PROVIDER == "openai"
    assert tree_manager.settings.COMPLEX_PROVIDER == "openai"
    assert tree_manager.config.style == custom_style
    assert tree_manager.config.agent_description == custom_agent_description
    assert tree_manager.config.end_goal == custom_end_goal


@pytest.mark.asyncio
async def test_load_config_tree():
    """
    Test loading a config for a tree from the Weaviate database.
    This should only update the tree settings, not the atlas or branch init.
    """
    user_manager = get_user_manager()
    user_id = "test_user_load_config_tree"
    config_id = "test_config_tree"
    conversation_id = "test_conversation_load_config_tree"
    config_name = "Test config tree"

    await initialise_user_and_tree(user_id, conversation_id)

    # Create a config to save
    settings = {
        "BASE_MODEL": "gpt-4-turbo",
        "COMPLEX_MODEL": "gpt-4-vision",
        "BASE_PROVIDER": "openai",
        "COMPLEX_PROVIDER": "openai",
    }
    custom_style = "Analytical and detailed."
    custom_agent_description = (
        "You are an analytical assistant providing detailed analyses."
    )
    custom_end_goal = "Provide thorough and analytical responses to queries."
    custom_branch_init = "empty"

    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": settings,
                "style": custom_style,
                "agent_description": custom_agent_description,
                "end_goal": custom_end_goal,
                "branch_initialisation": custom_branch_init,
            },
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Load the config for the tree (just settings, not atlas or branch init)
    response = await load_config_tree(
        user_id=user_id,
        conversation_id=conversation_id,
        config_id=config_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Verify the tree settings were updated but not the atlas or branch init
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tree.settings.BASE_MODEL == "gpt-4-turbo"
    assert tree.settings.COMPLEX_MODEL == "gpt-4-vision"
    assert tree.settings.BASE_PROVIDER == "openai"
    assert tree.settings.COMPLEX_PROVIDER == "openai"

    # Load everything now
    response = await load_config_tree(
        user_id=user_id,
        conversation_id=conversation_id,
        config_id=config_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Verify everything changed
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tree.settings.BASE_MODEL == "gpt-4-turbo"
    assert tree.settings.COMPLEX_MODEL == "gpt-4-vision"
    assert tree.tree_data.atlas.style == custom_style
    assert tree.tree_data.atlas.agent_description == custom_agent_description
    assert tree.tree_data.atlas.end_goal == custom_end_goal
    # assert tree.branch_initialisation == custom_branch_init

    # User level settings should remain unchanged
    user = await user_manager.get_user_local(user_id)
    tree_manager = user["tree_manager"]
    assert tree_manager.settings.BASE_MODEL == "gpt-4-turbo"
    assert tree_manager.settings.COMPLEX_MODEL == "gpt-4-vision"
    assert tree_manager.config.style == custom_style


@pytest.mark.asyncio
async def test_list_configs():
    """
    Test listing all configs for a user from the Weaviate database.
    """

    user_manager = get_user_manager()
    user_id = "test_user_list_configs"
    conversation_id = "test_conversation_list_configs"
    config_ids = [
        f"test_config_list_{i}_{random.randint(0, 1000000)}" for i in range(3)
    ]
    await initialise_user_and_tree(user_id, conversation_id)

    # Save multiple configs
    for i, config_id in enumerate(config_ids):
        config = SaveConfigUserData(
            name=f"Test config list {i}",
            default=False,
            config={
                "settings": {
                    "BASE_MODEL": f"model-{i}",
                    "COMPLEX_MODEL": f"complex-model-{i}",
                    "BASE_PROVIDER": "test-provider",
                    "COMPLEX_PROVIDER": "test-provider",
                },
                "style": f"Style {i}",
                "agent_description": f"Agent description {i}",
                "end_goal": f"End goal {i}",
                "branch_initialisation": "one_branch",
            },
            frontend_config={},
        )

        response = await save_config_user(
            user_id=user_id,
            config_id=config_id,
            data=config,
            user_manager=get_user_manager(),
        )
        response = read_response(response)
        assert response["error"] == ""

    response = await list_configs(
        user_id=user_id,
        user_manager=get_user_manager(),
    )
    response = read_response(response)
    assert response["error"] == ""

    for config_id in config_ids:
        assert config_id in [c["config_id"] for c in response["configs"]]


@pytest.mark.asyncio
async def test_config_save_load_workflow():
    """
    Test the config save/load workflow within a Weaviate database.
    """
    user_manager = get_user_manager()
    user_id_1 = "test_user_config_workflow_1"
    conversation_id_1 = "test_conversation_workflow_1"
    conversation_id_2 = "test_conversation_workflow_2"
    config_id = "test_shared_config_workflow"
    config_name = "Test shared config workflow"

    await initialise_user_and_tree(user_id_1, conversation_id_1)

    # Initialize the user with custom settings
    custom_settings = {
        "BASE_MODEL": "gpt-4o-mini",
        "COMPLEX_MODEL": "gpt-4o",
        "BASE_PROVIDER": "openai",
        "COMPLEX_PROVIDER": "openai",
    }
    custom_style = "Original style"
    custom_agent_description = "Original agent description"
    custom_end_goal = "Original end goal"

    # Apply the config
    user1 = await user_manager.get_user_local(user_id_1)
    tree_manager1: TreeManager = user1["tree_manager"]

    response = await save_config_user(
        user_id=user_id_1,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": tree_manager1.settings.to_json(),
                "style": custom_style,
                "agent_description": custom_agent_description,
                "end_goal": custom_end_goal,
                "branch_initialisation": tree_manager1.config.branch_initialisation,
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # update frontend config so that configs are saved to weaviate
    response = await save_config_user(
        user_id=user_id_1,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": custom_settings,
                "style": custom_style,
                "agent_description": custom_agent_description,
                "end_goal": custom_end_goal,
                "branch_initialisation": tree_manager1.config.branch_initialisation,
            },
            frontend_config={
                "save_location_wcd_url": os.getenv("WCD_URL"),
                "save_location_wcd_api_key": os.getenv("WCD_API_KEY"),
                "save_trees_to_weaviate": False,
                "save_configs_to_weaviate": True,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # list configs
    response = await list_configs(
        user_id=user_id_1,
        user_manager=user_manager,
    )
    list_response1 = read_response(response)
    assert config_id in [c["config_id"] for c in list_response1["configs"]]

    # change the config only in-memory
    response = await save_config_user(
        user_id=user_id_1,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": {
                    "BASE_MODEL": "claude-3-haiku",
                    "BASE_PROVIDER": "anthropic",
                },
                "style": "New incorrect style",
                "agent_description": "New incorrect agent description",
                "end_goal": "New incorrect end goal",
                "branch_initialisation": tree_manager1.config.branch_initialisation,
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # check the new config options are applied
    user1 = await user_manager.get_user_local(user_id_1)
    tree_manager1: TreeManager = user1["tree_manager"]
    assert tree_manager1.settings.BASE_MODEL == "claude-3-haiku"
    assert tree_manager1.settings.BASE_PROVIDER == "anthropic"
    assert tree_manager1.config.style == "New incorrect style"
    assert tree_manager1.config.agent_description == "New incorrect agent description"
    assert tree_manager1.config.end_goal == "New incorrect end goal"

    # Now load the config to the user
    response = await load_config_user(
        user_id=user_id_1,
        config_id=config_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Now verify the new settings were loaded
    user1 = await user_manager.get_user_local(user_id_1)
    tree_manager1: TreeManager = user1["tree_manager"]
    assert tree_manager1.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree_manager1.config.style == "Original style"
    assert tree_manager1.config.agent_description == "Original agent description"
    assert tree_manager1.config.end_goal == "Original end goal"

    user1_frontend_config = user1["frontend_config"]
    assert user1_frontend_config.save_location_wcd_url == os.getenv("WCD_URL")
    assert user1_frontend_config.save_location_wcd_api_key == os.getenv("WCD_API_KEY")
    assert user1_frontend_config.config["save_trees_to_weaviate"] == False

    # Make a new tree for this user
    response = await initialise_tree(
        user_id=user_id_1,
        conversation_id=conversation_id_2,
        data=InitialiseTreeData(
            low_memory=True,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # dont save
    response = await update_frontend_config(
        user_id=user_id_1,
        data=UpdateFrontendConfigData(
            config={"save_configs_to_weaviate": False},
        ),
        user_manager=user_manager,
    )

    # change the config globally for the user
    response = await save_config_user(
        user_id=user_id_1,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": {
                    "BASE_MODEL": "claude-3-5-sonnet",
                    "BASE_PROVIDER": "anthropic",
                },
                "style": "New updated style",
                "agent_description": "New updated agent description",
                "end_goal": "New updated end goal",
                "branch_initialisation": tree_manager1.config.branch_initialisation,
            },
            frontend_config={},
        ),
        user_manager=user_manager,
    )

    # Changes to user should propagate to the tree (not used its own settings)
    tree2: Tree = await user_manager.get_tree(user_id_1, conversation_id_2)
    assert tree2.settings.BASE_MODEL == "claude-3-5-sonnet"  # this one changes
    assert tree2.tree_data.atlas.style == "New updated style"
    assert tree2.tree_data.atlas.agent_description == "New updated agent description"
    assert tree2.tree_data.atlas.end_goal == "New updated end goal"

    # Load the config to the new tree 1 directly
    response = await load_config_tree(
        user_id=user_id_1,
        conversation_id=conversation_id_2,
        config_id=config_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Verify tree has the ORIGINAL settings (from the save/load)
    tree2 = await user_manager.get_tree(user_id_1, conversation_id_2)
    assert tree2.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree2.settings.COMPLEX_MODEL == "gpt-4o"
    assert tree2.tree_data.atlas.style == "Original style"
    assert tree2.tree_data.atlas.agent_description == "Original agent description"
    assert tree2.tree_data.atlas.end_goal == "Original end goal"


@pytest.mark.asyncio
async def test_delete_config():
    """
    Test deleting a config from the Weaviate database.
    """
    user_manager = get_user_manager()
    user_id = "test_user_delete_config"
    conversation_id = "test_conversation_delete_config"
    config_id = "test_delete_config"
    config_name = "Test delete config"
    await initialise_user_and_tree(user_id, conversation_id)

    # create a config and save
    user = await user_manager.get_user_local(user_id)
    tree_manager: TreeManager = user["tree_manager"]

    # change config
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=False,
            config={
                "settings": tree_manager.settings.to_json(),
                "style": tree_manager.config.style,
                "agent_description": tree_manager.config.agent_description,
                "end_goal": tree_manager.config.end_goal,
                "branch_initialisation": tree_manager.config.branch_initialisation,
            },
            frontend_config={},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # verify the config was saved by listing all configs
    response = await list_configs(
        user_id=user_id,
        user_manager=user_manager,
    )

    response = read_response(response)
    assert response["error"] == ""
    assert config_id in [c["config_id"] for c in response["configs"]]

    # delete the config
    response = await delete_config(
        user_id=user_id,
        config_id=config_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # verify the config was deleted by listing all configs
    response = await list_configs(
        user_id=user_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert config_id not in [c["config_id"] for c in response["configs"]]
