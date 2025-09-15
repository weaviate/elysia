import os
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from uuid import uuid4

load_dotenv(override=True)

# API Types
from elysia.api.api_types import SaveConfigTreeData
from elysia.api.utils.config import Config
from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.parsing import format_dict_to_serialisable
from elysia.tree.tree import Tree
from elysia.config import Settings, is_api_key

from weaviate.util import generate_uuid5


def rename_keys(config_item: dict):

    renamed_settings = {
        key.upper(): config_item["settings"][key] for key in config_item["settings"]
    }

    if "API_KEYS" in config_item["settings"]:
        renamed_api_keys = {
            key.lower(): config_item["settings"]["API_KEYS"][key]
            for key in config_item["settings"]["API_KEYS"]
        }
        renamed_settings["API_KEYS"] = renamed_api_keys

    renamed_config = {key.lower(): config_item[key] for key in config_item}
    renamed_config["settings"] = renamed_settings

    return renamed_config


router = APIRouter()


@router.get("/{user_id}/{conversation_id}")
async def get_tree_config(
    user_id: str,
    conversation_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get the config for a single tree / conversation.
    """
    logger.debug(f"/get_tree_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Conversation ID: {conversation_id}")

    headers = {"Cache-Control": "no-cache"}

    try:
        tree: Tree = await user_manager.get_tree(user_id, conversation_id)
        config = Config(
            id="tree_config",
            name="Tree Config",
            settings=tree.settings,
            style=tree.tree_data.atlas.style,
            agent_description=tree.tree_data.atlas.agent_description,
            end_goal=tree.tree_data.atlas.end_goal,
            branch_initialisation=tree.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /get_tree_config API")
        return JSONResponse(content={"error": str(e), "config": {}}, headers=headers)

    return JSONResponse(
        content={"error": "", "config": config.to_json()}, headers=headers
    )


@router.post("/{user_id}/{conversation_id}/new")
async def new_tree_config(
    user_id: str,
    conversation_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Set the config in a single tree / conversation to the default values (gemini 2.0 flash or environment variables).
    Any changes will only affect this one tree / conversation.

    Args:
        user_id (str): The user ID.
        conversation_id (str): The conversation ID.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/new_tree_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Conversation ID: {conversation_id}")

    try:
        tree: Tree = await user_manager.get_tree(user_id, conversation_id)

        settings = Settings()
        settings.smart_setup()

        tree.configure(
            **{k: v for k, v in settings.to_json().items() if k != "API_KEYS"}
        )
        tree.configure(**settings.to_json()["API_KEYS"])

        tree.change_style("Informative, polite and friendly.")
        tree.change_agent_description(
            "You search and query Weaviate to satisfy the user's query, providing a concise summary of the results."
        )
        tree.change_end_goal(
            "You have satisfied the user's query, and provided a concise summary of the results. "
            "Or, you have exhausted all options available, or asked the user for clarification."
        )
        tree.set_branch_initialisation("one_branch")

        config = Config(
            id="tree_config",
            name="Tree Config",
            settings=tree.settings,
            style=tree.tree_data.atlas.style,
            agent_description=tree.tree_data.atlas.agent_description,
            end_goal=tree.tree_data.atlas.end_goal,
            branch_initialisation=tree.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /new_tree_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.to_json()})


@router.post("/{user_id}/{conversation_id}")
async def change_config_tree(
    user_id: str,
    conversation_id: str,
    data: SaveConfigTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Change the config for a single tree / conversation.
    The `config` dictionary can contain any of the settings (dict), style, agent_description, end_goal and branch_initialisation.
    You do not need to specify all the settings, only the ones you wish to change.

    Once you change the config, the tree will no longer inherit from the user's config.

    In this case, you cannot change the API keys, it will always inherit from the user's config.
    Any API keys provided will be ignored.

    Args:
        user_id (str): Required. The user ID.
        conversation_id (str): Required. The conversation ID.
        data (Config): Required. Containing the following attributes:
            settings(dict): Optional. A dictionary of config values to set, follows the same format as the Settings.configure() method.
            style (str): Optional. The writing style of the agent.
            agent_description (str): Optional. The description of the agent.
            end_goal (str): Optional. The end goal of the agent.
            branch_initialisation (str): Optional. The branch initialisation of the agent.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/change_config_tree API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Conversation ID: {conversation_id}")
    logger.debug(f"Settings: {data.settings}")
    logger.debug(f"Style: {data.style}")
    logger.debug(f"Agent description: {data.agent_description}")
    logger.debug(f"End goal: {data.end_goal}")
    logger.debug(f"Branch initialisation: {data.branch_initialisation}")

    try:
        tree: Tree = await user_manager.get_tree(user_id, conversation_id)

        if data.settings is not None:

            # tree level configs do not include API keys
            if "wcd_url" in data.settings:
                del data.settings["wcd_url"]

            api_keys = [key for key in data.settings if is_api_key(key)]
            for key in api_keys:
                del data.settings[key]

            tree.configure(**data.settings)

        if data.style is not None:
            tree.change_style(data.style)

        if data.agent_description is not None:
            tree.change_agent_description(data.agent_description)

        if data.end_goal is not None:
            tree.change_end_goal(data.end_goal)

        if data.branch_initialisation is not None:
            tree.set_branch_initialisation(data.branch_initialisation)

        config = Config(
            id="tree_config",
            name="Tree Config",
            settings=tree.settings,
            style=tree.tree_data.atlas.style,
            agent_description=tree.tree_data.atlas.agent_description,
            end_goal=tree.tree_data.atlas.end_goal,
            branch_initialisation=tree.branch_initialisation,
        )

    except Exception as e:
        logger.exception(f"Error in /change_config_tree API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": config.to_json()})


@router.post("/{user_id}/{conversation_id}/{config_id}/load")
async def load_config_tree(
    user_id: str,
    conversation_id: str,
    config_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Load a config from the weaviate database for a single tree / conversation.
    This will set the config for the current tree.
    It will not affect other trees associated with the user.
    It will NOT update the ClientManager to use the new API keys.

    Args:
        user_id (str): The user ID.
        conversation_id (str): The conversation ID.
        config_id (str): The ID of the config to load. Can be found in the /list_configs API.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/load_config_tree API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Conversation ID: {conversation_id}")
    logger.debug(f"Config ID: {config_id}")

    try:
        # Retrieve the user info
        user = await user_manager.get_user_local(user_id)
        tree: Tree = await user_manager.get_tree(user_id, conversation_id)

        if (
            "frontend_config" not in user
            or "save_location_wcd_url" not in user["frontend_config"].__dict__
            or "save_location_wcd_api_key" not in user["frontend_config"].__dict__
        ):
            raise Exception("WCD URL or API key not found.")

        if (
            user["frontend_config"].save_location_wcd_url is None
            or user["frontend_config"].save_location_wcd_api_key is None
        ):
            raise Exception(
                "No valid destination for config load location found. "
                "Please update the save location using the /update_save_location API."
            )

        # Retrieve the config from the weaviate database
        async with user[
            "frontend_config"
        ].save_location_client_manager.connect_to_async_client() as client:
            uuid = generate_uuid5(config_id)
            collection = client.collections.get("ELYSIA_CONFIG__")
            config_item = await collection.query.fetch_object_by_id(uuid=uuid)

        # Load the configs to the current user
        format_dict_to_serialisable(config_item.properties)
        renamed_config = rename_keys(config_item.properties)

        tree.configure(
            **{
                c: renamed_config["settings"][c]
                for c in renamed_config["settings"]
                if (c != "api_keys" and not is_api_key(c))
            }
        )

        tree.change_style(renamed_config["style"])
        tree.change_agent_description(renamed_config["agent_description"])
        tree.change_end_goal(renamed_config["end_goal"])

        tree.set_branch_initialisation(renamed_config["branch_initialisation"])

    except Exception as e:
        logger.exception(f"Error in /load_config API")
        return JSONResponse(content={"error": str(e), "config": {}})

    return JSONResponse(content={"error": "", "config": renamed_config})
