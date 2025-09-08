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


def get_frontend_config_file_path(user_id: str) -> Path:
    elysia_package_dir = Path(__file__).parent.parent.parent.parent  # Gets to elysia/
    config_dir = elysia_package_dir / "elysia" / "api" / "user_configs"
    config_file = config_dir / f"frontend_config_{user_id}.json"
    return config_file


async def load_frontend_config_from_file(
    user_id: str, logger: Logger
) -> FrontendConfig:
    try:
        elysia_package_dir = Path(
            __file__
        ).parent.parent.parent.parent  # Gets to elysia/
        config_dir = elysia_package_dir / "elysia" / "api" / "user_configs"
        config_file = config_dir / f"frontend_config_{user_id}.json"

        if not config_file.exists():
            return FrontendConfig(logger=logger)

        with open(config_file, "r") as f:
            fe_config = await FrontendConfig.from_json(json.load(f), logger)
            return fe_config

    except Exception as e:
        logger.error(f"Error in save_frontend_config_to_file: {str(e)}")
        return FrontendConfig(logger=logger)


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
async def test_get_models():
    """
    Test the endpoint for getting the models.
    """
    user_manager = get_user_manager()
    response = await get_models()
    response = read_response(response)
    assert response["error"] == ""
    assert "models" in response
    assert len(response["models"]) > 0
    assert "openai" in response["models"]
    assert "anthropic" in response["models"]
    assert "gemini" in response["models"]


@pytest.mark.asyncio
async def test_config_workflow():
    """
    Test the config workflow.
    """
    user_manager = get_user_manager()
    user_id = "test_user_config_workflow"
    conversation_id_1 = "test_conversation_workflow_1"
    conversation_id_2 = "test_conversation_workflow_2"
    config_id = f"test_config_conversation_config_workflow"
    config_name = "Test config workflow"

    await initialise_user_and_tree(user_id, conversation_id_1)

    # Test changing config locally
    user_settings = {
        "BASE_MODEL": "gpt-4o-mini",
        "COMPLEX_MODEL": "gpt-4o",
        "BASE_PROVIDER": "openai",
        "COMPLEX_PROVIDER": "openai",
    }
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=False,
            config={
                "settings": user_settings,
                "style": "Professional style for user",
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Check tree manager settings have NOT changed because default=False
    tree_manager = user_manager.users[user_id]["tree_manager"]
    assert tree_manager.settings.BASE_MODEL != "gpt-4o-mini"
    assert tree_manager.settings.COMPLEX_MODEL != "gpt-4o"
    assert tree_manager.settings.BASE_PROVIDER != "openai"
    assert tree_manager.settings.COMPLEX_PROVIDER != "openai"

    # save again with default=True
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": user_settings,
                "style": "Professional style for user",
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    assert tree_manager.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree_manager.settings.COMPLEX_MODEL == "gpt-4o"
    assert tree_manager.settings.BASE_PROVIDER == "openai"
    assert tree_manager.settings.COMPLEX_PROVIDER == "openai"

    # Verify tree 1 inherits user model settings including style
    tree1: Tree = await user_manager.get_tree(user_id, conversation_id_1)
    assert tree1.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree1.settings.COMPLEX_MODEL == "gpt-4o"
    assert tree1.tree_data.atlas.style == "Professional style for user"

    # Create tree 2
    response = await initialise_tree(
        user_id,
        conversation_id_2,
        data=InitialiseTreeData(
            low_memory=True,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Tree 2 should inherit the new settings
    tree2: Tree = await user_manager.get_tree(user_id, conversation_id_2)
    assert tree2.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree2.settings.COMPLEX_MODEL == "gpt-4o"

    # Change tree 1 settings
    tree1_settings = {
        "BASE_MODEL": "claude-3-haiku",
        "COMPLEX_MODEL": "claude-3-opus",
        "BASE_PROVIDER": "anthropic",
        "COMPLEX_PROVIDER": "anthropic",
    }
    response = await change_config_tree(
        user_id=user_id,
        conversation_id=conversation_id_1,
        data=SaveConfigTreeData(
            settings=tree1_settings,
            style="Custom tree 1 style",
            agent_description="Custom tree 1 agent description",
            end_goal="Custom tree 1 end goal",
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Verify tree 1 has its own settings now
    tree1 = await user_manager.get_tree(user_id, conversation_id_1)
    assert tree1.settings.BASE_MODEL == "claude-3-haiku"
    assert tree1.settings.COMPLEX_MODEL == "claude-3-opus"
    assert tree1.tree_data.atlas.style == "Custom tree 1 style"

    # change user settings again
    new_user_settings = {
        "BASE_MODEL": "gemini-pro",
        "COMPLEX_MODEL": "gemini-ultra",
        "BASE_PROVIDER": "google",
        "COMPLEX_PROVIDER": "google",
    }
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": new_user_settings,
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Tree 1 should keep its own settings
    tree1 = await user_manager.get_tree(user_id, conversation_id_1)
    assert tree1.settings.BASE_MODEL == "claude-3-haiku"
    assert tree1.settings.COMPLEX_MODEL == "claude-3-opus"

    # But tree 2 should inherit updated user settings
    tree2 = await user_manager.get_tree(user_id, conversation_id_2)
    assert tree2.settings.BASE_MODEL == "gemini-pro"
    assert tree2.settings.COMPLEX_MODEL == "gemini-ultra"


@pytest.mark.asyncio
async def test_change_config_tree():
    """
    Test the endpoint for changing the config of a tree.
    """
    user_manager = get_user_manager()
    user_id = "test_user_change_config_tree"
    conversation_id = "test_conversation_change_config_tree"

    await initialise_user_and_tree(user_id, conversation_id)

    # get initial settings
    tree: Tree = await user_manager.get_tree(user_id, conversation_id)
    initial_base_model = tree.settings.BASE_MODEL
    initial_style = tree.tree_data.atlas.style

    # change tree config
    new_settings = {
        "BASE_MODEL": "gpt-3.5-turbo",
        "COMPLEX_MODEL": "gpt-4-turbo",
        "BASE_PROVIDER": "openai",
        "COMPLEX_PROVIDER": "openai",
        # Try including API keys, which should be ignored
        "openai_api_key": "test_key_should_be_ignored",
        "wcd_url": "should_be_ignored_too",
    }
    new_style = "Technical and detailed."
    new_agent_description = "You are a technical assistant."
    new_end_goal = "Provide technical details to the user."
    new_branch_init = "empty"

    response = await change_config_tree(
        user_id=user_id,
        conversation_id=conversation_id,
        data=SaveConfigTreeData(
            settings=new_settings,
            style=new_style,
            agent_description=new_agent_description,
            end_goal=new_end_goal,
            branch_initialisation=new_branch_init,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # Tree settings should have been updated
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tree.settings.BASE_MODEL == "gpt-3.5-turbo"
    assert tree.settings.COMPLEX_MODEL == "gpt-4-turbo"
    assert tree.settings.BASE_PROVIDER == "openai"
    assert tree.settings.COMPLEX_PROVIDER == "openai"
    assert tree.tree_data.atlas.style == new_style
    assert tree.tree_data.atlas.agent_description == new_agent_description
    assert tree.tree_data.atlas.end_goal == new_end_goal
    # assert tree.branch_initialisation == new_branch_init

    # But not api keys
    assert (
        "openai_api_key" not in tree.settings.to_json()
        or tree.settings.to_json()["openai_api_key"] != "test_key_should_be_ignored"
    )
    assert tree.settings.WCD_URL != "should_be_ignored_too"

    # user-level settings should remain unchanged
    user = await user_manager.get_user_local(user_id)
    tree_manager = user["tree_manager"]
    assert tree_manager.settings.BASE_MODEL == initial_base_model
    assert tree_manager.config.style == initial_style


@pytest.mark.asyncio
async def test_new_config():
    """
    Test the endpoint for creating a new config.
    """
    user_manager = get_user_manager()
    user_id = "test_user_new_config"
    conversation_id = "test_conversation_new_config"
    config_id = f"test_new_config"
    config_name = "Test new config"

    await initialise_user_and_tree(user_id, conversation_id)

    # create a new config
    response = await new_user_config(
        user_id=user_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]
    new_user_config_id = response["config"]["id"]

    # save config
    response = await save_config_user(
        user_id=user_id,
        config_id=new_user_config_id,
        data=SaveConfigUserData(
            name=config_name,
            id=new_user_config_id,
            default=True,
            config=response["config"],
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]

    response = await initialise_user(
        user_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert response["config"]["id"] == new_user_config_id


@pytest.mark.asyncio
async def test_save_config_no_api_keys():
    """
    Test that a config can be saved with no api keys.
    """
    user_manager = get_user_manager()
    user_id = "test_user_save_config_no_api_keys"
    conversation_id = "test_conversation_save_config_no_api_keys"
    config_id = f"test_config_conversation_save_config_no_api_keys"
    config_name = "Test save config no api keys"

    await initialise_user_and_tree(user_id, conversation_id)

    # Get tree manager in advance
    user = await user_manager.get_user_local(user_id)
    tree_manager = user["tree_manager"]

    # Save a config with no api keys
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": {
                    "BASE_MODEL": "gpt-4o-mini",
                    "COMPLEX_MODEL": "gpt-4o",
                    "BASE_PROVIDER": "openai",
                    "COMPLEX_PROVIDER": "openai",
                    "API_KEYS": {},
                },
                "style": "Professional and concise.",
                "agent_description": "You are a helpful assistant that searches data.",
                "end_goal": "Provide a summary of the data.",
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


@pytest.mark.asyncio
async def test_change_config_user():
    """
    Test changing a user config propagates to the TreeManager and the trees.
    """
    user_manager = get_user_manager()
    user_id = "test_user_change_config_user"
    conversation_id = "test_conversation_change_config_user"
    config_id = f"test_config_conversation_change_config_user"
    config_name = "Test change config user"

    await initialise_user_and_tree(user_id, conversation_id)

    # Get tree manager in advance
    user = await user_manager.get_user_local(user_id)
    tree_manager = user["tree_manager"]

    # Set frontend config to not save configs to weaviate
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={},
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert response["warnings"] == [], "\n".join(response["warnings"])

    # Change user config
    new_settings = {
        "BASE_MODEL": "gpt-4o-mini",
        "COMPLEX_MODEL": "gpt-4o",
        "BASE_PROVIDER": "openai",
        "COMPLEX_PROVIDER": "openai",
    }
    new_style = "Professional and concise."
    new_agent_description = "You are a helpful assistant that searches data."
    new_branch_init = "one_branch"
    old_end_goal = tree_manager.config.end_goal

    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            default=True,
            config={
                "settings": new_settings,
                "style": new_style,
                "agent_description": new_agent_description,
                "branch_initialisation": new_branch_init,
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )

    # Verify response is OK
    response = read_response(response)
    assert response["error"] == "", response["error"]
    assert response["warnings"] == [], "\n".join(response["warnings"])

    assert tree_manager.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree_manager.settings.COMPLEX_MODEL == "gpt-4o"
    assert tree_manager.settings.BASE_PROVIDER == "openai"
    assert tree_manager.settings.COMPLEX_PROVIDER == "openai"

    assert tree_manager.config.id == config_id
    assert tree_manager.config.style == new_style
    assert tree_manager.config.agent_description == new_agent_description
    assert (
        tree_manager.config.end_goal == old_end_goal
    )  # this parameter wasn't set so it should be the same as before
    assert tree_manager.config.branch_initialisation == new_branch_init

    # Verify model settings propagated to existing tree
    tree: Tree = await user_manager.get_tree(user_id, conversation_id)
    assert tree.settings.BASE_MODEL == "gpt-4o-mini"
    assert tree.settings.COMPLEX_MODEL == "gpt-4o"
    assert tree.settings.BASE_PROVIDER == "openai"
    assert tree.settings.COMPLEX_PROVIDER == "openai"

    # And style, agent_description, etc. should have propagated to existing tree
    assert tree.tree_data.atlas.style == new_style
    assert tree.tree_data.atlas.agent_description == new_agent_description
    assert tree.branch_initialisation == new_branch_init


@pytest.mark.asyncio
@pytest.mark.skip(reason="Skipping frontend config local test")
async def test_frontend_config_local():
    """
    Test that the frontend config is saved to a local file and loads correctly.
    """
    user_manager = get_user_manager()
    user_id = "test_user_frontend_config_local"
    conversation_id = "test_conversation_frontend_config_local"
    config_id = f"test_frontend_config_local"
    config_name = "Test frontend config local"

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
            default=True,
            config={},
            frontend_config={
                "save_trees_to_weaviate": False,
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)

    # check existence of local file
    # assert os.path.exists(get_frontend_config_file_path(user_id))
    # skip for now

    # check the local file has the correct values
    fe_config = await load_frontend_config_from_file(user_id, logger)
    fe_config = fe_config.to_json()

    assert fe_config["save_trees_to_weaviate"] == False
    assert fe_config["save_configs_to_weaviate"] == False

    # remove user from user manager
    del user_manager.users[user_id]

    # add user back to user manager
    await user_manager.add_user_local(user_id)

    # check the frontend config is loaded from the local file and correct settings
    user = await user_manager.get_user_local(user_id)
    assert user["frontend_config"].config["save_trees_to_weaviate"] == False
    assert user["frontend_config"].config["save_configs_to_weaviate"] == False


@pytest.mark.asyncio
async def test_new_config_no_overwrite():
    """
    Test creating a new config does not overwrite existing config.
    """
    user_manager = get_user_manager()
    user_id = "test_user_new_config_no_overwrite"
    conversation_id = "test_conversation_new_config_no_overwrite"
    config_name = "Test new config no overwrite"

    await initialise_user_and_tree(user_id, conversation_id)

    # get config id
    user = await user_manager.get_user_local(user_id)
    config_id = user["tree_manager"].config.id

    # save some values
    response = await save_config_user(
        user_id=user_id,
        config_id=config_id,
        data=SaveConfigUserData(
            name=config_name,
            id=config_id,
            default=True,
            config={
                "settings": {
                    "BASE_MODEL": "gpt-4o-mini",
                    "BASE_PROVIDER": "openai",
                    "COMPLEX_MODEL": "gpt-4o",
                    "COMPLEX_PROVIDER": "openai",
                    "API_KEYS": {
                        "openai_api_key": "some-random-openai-api-key",
                    },
                },
                "style": "New style for no overwrite",
                "agent_description": "New agent description for no overwrite",
                "end_goal": "New end goal for no overwrite",
                "branch_initialisation": "one_branch",
            },
            frontend_config={
                "save_trees_to_weaviate": False,
                "save_configs_to_weaviate": False,
            },
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # create a new config
    response = await new_user_config(
        user_id=user_id,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    new_config_id = response["config"]["id"]

    # check the user values were NOT saved
    user = await user_manager.get_user_local(user_id)
    assert user["tree_manager"].config.id == config_id
    assert user["tree_manager"].config.settings.BASE_MODEL == "gpt-4o-mini"
    assert user["tree_manager"].config.settings.BASE_PROVIDER == "openai"
    assert user["tree_manager"].config.settings.COMPLEX_MODEL == "gpt-4o"
    assert user["tree_manager"].config.settings.COMPLEX_PROVIDER == "openai"
    assert "openai_api_key" in user["tree_manager"].config.settings.API_KEYS
    assert (
        user["tree_manager"].config.settings.API_KEYS["openai_api_key"]
        == "some-random-openai-api-key"
    )
