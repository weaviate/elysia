import os
from cryptography.fernet import InvalidToken
import json
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv, set_key

load_dotenv(override=True)

from elysia.api.api_types import (
    SaveConfigUserData,
    UpdateFrontendConfigData,
)
from elysia.api.utils.config import Config
from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.util.parsing import format_dict_to_serialisable, format_datetime
from elysia.config import Settings
from elysia.api.services.tree import TreeManager
from elysia.api.utils.config import FrontendConfig
from elysia.api.utils.encryption import encrypt_api_keys, decrypt_api_keys
from elysia.api.utils.models import models

import weaviate.classes.config as wc
from weaviate.util import generate_uuid5
from weaviate.classes.query import MetadataQuery, Sort, Filter
from weaviate.classes.config import Property, DataType
from uuid import uuid4


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


@router.get("/models")
async def get_models():
    try:
        return JSONResponse(content={"models": models, "error": ""})
    except Exception as e:
        logger.exception(f"Error in /get_models API")
        return JSONResponse(content={"models": {}, "error": str(e)})


@router.patch("/frontend_config/{user_id}")
async def update_frontend_config(
    user_id: str,
    data: UpdateFrontendConfigData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Update a user's frontend config, e.g. whether to save trees to weaviate or not, and the save location.
    """
    logger.debug(f"/update_frontend_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config: {data.config}")

    try:
        await user_manager.update_frontend_config(user_id, data.config)
        return JSONResponse(content={"error": ""})
    except Exception as e:

        if "Could not connect to Weaviate" in str(e):
            return JSONResponse(
                content={
                    "error": "Could not connect to Weaviate. Double check that the WCD_URL and WCD_API_KEY are correct."
                }
            )
        else:
            logger.exception(f"Error in /update_frontend_config API")
            return JSONResponse(content={"error": str(e)})


@router.get("/{user_id}")
async def get_current_user_config(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get the current config for a single user (in memory).
    """
    logger.debug(f"/get_current_user_config API request received")
    logger.debug(f"User ID: {user_id}")

    headers = {"Cache-Control": "no-cache"}

    try:
        user = await user_manager.get_user_local(user_id)
        config = user["tree_manager"].config.to_json()
        frontend_config = user["frontend_config"].to_json()

    except Exception as e:
        logger.exception(f"Error in /get_current_user_config API")
        return JSONResponse(
            content={"error": str(e), "config": {}, "frontend_config": {}},
            headers=headers,
        )

    return JSONResponse(
        content={
            "error": "",
            "config": config,
            "frontend_config": frontend_config,
        },
        headers=headers,
    )


@router.get("/{user_id}/{config_id}/load")
async def load_a_config(
    user_id: str,
    config_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Get the current config for a single user (in memory).
    """
    logger.debug(f"/get_current_user_config API request received")
    logger.debug(f"User ID: {user_id}")

    headers = {"Cache-Control": "no-cache"}
    warnings = []

    try:
        user = await user_manager.get_user_local(user_id)
        frontend_config = user["frontend_config"]

        # check if the user has a valid save location
        if (
            frontend_config.save_location_wcd_url == ""
            or frontend_config.save_location_wcd_api_key == ""
        ):
            raise Exception("WCD URL or API key not found.")

        if (
            frontend_config.save_location_wcd_url == ""
            or frontend_config.save_location_wcd_api_key == ""
        ):
            raise Exception(
                "No valid destination for config load location found. "
                "Please update the save location using the /update_save_location API."
            )

        # Retrieve the config from the weaviate database
        async with (
            frontend_config.save_location_client_manager.connect_to_async_client() as client
        ):
            uuid = generate_uuid5(config_id)
            collection = client.collections.get("ELYSIA_CONFIG__")
            config_response = await collection.query.fetch_object_by_id(uuid=uuid)
            if config_response is None:
                raise Exception(f"Config with ID {config_id} not found.")

            config_item: dict = config_response.properties  # type: ignore

        format_dict_to_serialisable(config_item)

        if "settings" in config_item:
            try:
                if "API_KEYS" in config_item["settings"]:
                    if "null" in config_item["settings"]["API_KEYS"]:
                        del config_item["settings"]["API_KEYS"]["null"]

                config_item["settings"] = decrypt_api_keys(config_item["settings"])
            except InvalidToken:
                warnings.append(
                    "Invalid token for config. You will need to re-enter your API keys in the settings."
                )
                config_item["settings"]["API_KEYS"] = {}
                config_item["settings"]["WCD_API_KEY"] = ""

        # Rename the keys to the correct format
        renamed_config = rename_keys(config_item)
        if "config_id" in renamed_config:
            renamed_config["id"] = renamed_config["config_id"]
            del renamed_config["config_id"]

    except Exception as e:
        logger.exception(f"Error in /get_current_user_config API")
        return JSONResponse(
            content={"error": str(e), "config": {}, "frontend_config": {}},
            headers=headers,
        )

    return JSONResponse(
        content={
            "error": "",
            "config": renamed_config,
            "frontend_config": frontend_config.to_json(),
            "warnings": warnings,
        },
        headers=headers,
    )


@router.post("/{user_id}/new")
async def new_user_config(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Set the base_model and complex_model to the default values (gemini 2.0 flash or environment variables), for an entire user.
    Any changes will propagate to all trees associated with the user that were not initialised with their own settings.

    Args:
        user_id (str): The user ID.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the new config values.
    """
    logger.debug(f"/new_user_config API request received")
    logger.debug(f"User ID: {user_id}")

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        frontend_config: FrontendConfig = user["frontend_config"]

        # get existing config

        settings = Settings()
        settings.smart_setup()

        config = Config(
            id=str(uuid4()),
            name="New Config",
            settings=settings,
            style="Informative, polite and friendly.",
            agent_description="You search and query Weaviate to satisfy the user's query, providing a concise summary of the results.",
            end_goal=(
                "You have satisfied the user's query, and provided a concise summary of the results. "
                "Or, you have exhausted all options available, or asked the user for clarification."
            ),
            branch_initialisation="one_branch",
        )

        frontend_config.save_location_wcd_url = frontend_config.save_location_wcd_url
        frontend_config.save_location_wcd_api_key = (
            frontend_config.save_location_wcd_api_key
        )
        frontend_config.config = {
            "save_trees_to_weaviate": True,
            "save_configs_to_weaviate": True,
            "client_timeout": int(os.getenv("CLIENT_TIMEOUT", 3)),
            "tree_timeout": int(os.getenv("TREE_TIMEOUT", 10)),
        }

    except Exception as e:
        logger.exception(f"Error in /new_user_config API")
        return JSONResponse(
            content={
                "error": str(e),
                "config": {},
                "frontend_config": {},
            }
        )

    return JSONResponse(
        content={
            "error": "",
            "config": config.to_json(),
            "frontend_config": user["frontend_config"].to_json(),
        }
    )


def save_frontend_config_to_file(user_id: str, frontend_config: dict):

    # save frontend config locally
    try:
        elysia_package_dir = Path(__file__).parent.parent.parent  # Gets to elysia/
        config_dir = elysia_package_dir / "api" / "user_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / f"frontend_config_{user_id}.json"

        with open(config_file, "w") as f:
            json.dump(frontend_config, f)
        logger.debug(f"Frontend config saved to local file: {config_file}")
    except Exception as e:
        logger.error(f"Error in save_frontend_config_to_file: {str(e)}")
        pass


@router.post("/{user_id}/{config_id}")
async def save_config_user(
    user_id: str,
    config_id: str,
    data: SaveConfigUserData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Change the global config for a user.
    The `config` dictionary can contain any of the `settings` (dict), style, agent_description, end_goal and branch_initialisation.
    You do not need to specify all the `settings`, only the ones you wish to change.

    Any changes to the _settings_ will propagate to all trees associated with the user that were not initialised with their own settings,
    as well as update the ClientManager to use the new API keys.
    However, changes to the _style_, _agent_description_, _end_goal_ and _branch_initialisation_ will not be changed for existing trees.
    I.e., only changes to API keys and model choices will be changed for existing trees (Settings).

    Additionally saves a config to the weaviate database.
    If the collection ELYSIA_CONFIG__ does not exist, it will be created.
    Uses the WCD_URL and WCD_API_KEY from /update_save_location to connect to the weaviate database.
    (These are initialised to the environment variables).

    Args:
        user_id (str): Required. The user ID.
        data (SaveConfigUserData): Required. Containing the following attributes:
            name (str): Optional. The name of the config.
            settings (dict): Optional. A dictionary of config values to set, follows the same format as the Settings.configure() method.
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
    logger.debug(f"/save_config_user API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {config_id}")
    logger.debug(f"Name: {data.name}")
    logger.debug(f"Backend Config: {data.config}")
    logger.debug(f"Frontend config: {data.frontend_config}")
    logger.debug(f"Default: {data.default}")

    warnings = []

    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
    except Exception as e:
        logger.exception(f"Error in /save_config_user API during user retrieval")
        return JSONResponse(
            content={"error": str(e), "config": {}, "frontend_config": {}}
        )

    if data.default:

        # Save the config in memory for this user (Apply step)
        try:
            await user_manager.update_config(
                user_id,
                conversation_id=None,
                config_id=config_id,
                config_name=data.name,
                settings=data.config.get("settings"),
                style=data.config.get("style"),
                agent_description=data.config.get("agent_description"),
                end_goal=data.config.get("end_goal"),
                branch_initialisation=data.config.get("branch_initialisation"),
            )

        except Exception as e:
            logger.exception(f"Error in /save_config_user API during in-memory update")
            return JSONResponse(
                content={
                    "error": str(e),
                    "config": tree_manager.config.to_json(),
                    "frontend_config": user["frontend_config"].to_json(),
                    "warnings": warnings,
                }
            )

        # Save the frontend config to memory and local file system
        if data.frontend_config is not None:
            try:
                await user_manager.update_frontend_config(user_id, data.frontend_config)
                save_frontend_config_to_file(user_id, user["frontend_config"].to_json())

            except Exception as e:
                if "Could not connect to Weaviate" in str(e):
                    warnings.append(
                        "Could not connect to Weaviate. "
                        "Please check your WCD_URL and WCD_API_KEY are correct."
                    )
                    return JSONResponse(
                        content={
                            "error": "",
                            "config": tree_manager.config.to_json(),
                            "frontend_config": user["frontend_config"].to_json(),
                            "warnings": warnings,
                        }
                    )
                else:
                    logger.exception(
                        f"Error in /save_config_user API during frontend config update"
                    )
                    return JSONResponse(
                        content={
                            "error": str(e),
                            "config": tree_manager.config.to_json(),
                            "frontend_config": user["frontend_config"].to_json(),
                            "warnings": warnings,
                        }
                    )

    if not user["frontend_config"].config["save_configs_to_weaviate"]:
        return JSONResponse(
            content={
                "error": "",
                "config": tree_manager.config.to_json(),
                "frontend_config": user["frontend_config"].to_json(),
                "warnings": warnings,
            }
        )

    # Save the config to the weaviate database
    try:

        # Retrieve the user info
        user = await user_manager.get_user_local(user_id)
        settings: Settings = user["tree_manager"].settings
        settings_dict = settings.to_json()

        # extract style, agent_description, end_goal, branch_initialisation from user
        style = user["tree_manager"].config.style
        agent_description = user["tree_manager"].config.agent_description
        end_goal = user["tree_manager"].config.end_goal
        branch_initialisation = user["tree_manager"].config.branch_initialisation

        # Check if the user has a valid save location
        if (
            user["frontend_config"].save_location_wcd_url == ""
            or user["frontend_config"].save_location_wcd_api_key == ""
        ):
            warnings.append(
                "No valid destination for config save location found. "
                "Config has not been saved to Weaviate. "
                "Saving to Weaviate requires a Config Storage WCD_URL and WCD_API_KEY."
            )
        else:

            settings_dict = encrypt_api_keys(settings_dict)

            # Connect to the weaviate database
            async with user[
                "frontend_config"
            ].save_location_client_manager.connect_to_async_client() as client:

                uuid = generate_uuid5(config_id)

                # Create a collection if it doesn't exist
                if await client.collections.exists("ELYSIA_CONFIG__"):
                    collection = client.collections.get("ELYSIA_CONFIG__")
                else:
                    collection = await client.collections.create(
                        "ELYSIA_CONFIG__",
                        vectorizer_config=wc.Configure.Vectorizer.none(),
                        inverted_index_config=wc.Configure.inverted_index(
                            index_timestamps=True
                        ),
                        properties=[
                            Property(
                                name="name",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="style",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="agent_description",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="end_goal",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="branch_initialisation",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="user_id",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="config_id",
                                data_type=DataType.TEXT,
                            ),
                            Property(
                                name="default",
                                data_type=DataType.BOOL,
                            ),
                        ],
                    )

                format_dict_to_serialisable(settings_dict)

                if settings_dict == {}:
                    settings_dict["null"] = "null"

                if "API_KEYS" in settings_dict and settings_dict["API_KEYS"] == {}:
                    settings_dict["API_KEYS"]["null"] = "null"

                config_item = {
                    "name": data.name,
                    "settings": settings_dict,
                    "style": style,
                    "agent_description": agent_description,
                    "end_goal": end_goal,
                    "branch_initialisation": branch_initialisation,
                    "frontend_config": user["frontend_config"].config,
                    "config_id": config_id,
                    "user_id": user_id,
                    "default": data.default,
                }

                # if the config is a default config, set all other default configs to False
                if data.default:
                    existing_default_config = await collection.query.fetch_objects(
                        filters=Filter.all_of(
                            [
                                Filter.by_property("default").equal(True),
                                Filter.by_property("user_id").equal(user_id),
                            ]
                        )
                    )
                    for item in existing_default_config.objects:
                        await collection.data.update(
                            properties={"default": False}, uuid=item.uuid
                        )

                # save the config to the weaviate database
                if await collection.data.exists(uuid=uuid):
                    await collection.data.update(properties=config_item, uuid=uuid)
                else:
                    await collection.data.insert(config_item, uuid=uuid)

    except Exception as e:
        logger.exception(f"Error in /save_config_user API during weaviate save")
        return JSONResponse(
            content={
                "error": str(e),
                "config": tree_manager.config.to_json(),
                "frontend_config": user["frontend_config"].to_json(),
                "warnings": warnings,
            }
        )

    logger.debug(f"Backend config returned: {tree_manager.config.to_json()}")
    logger.debug(f"Frontend config returned: {user['frontend_config'].to_json()}")

    return JSONResponse(
        content={
            "error": "",
            "config": tree_manager.config.to_json(),
            "frontend_config": user["frontend_config"].to_json(),
            "warnings": warnings,
        }
    )


@router.post("/{user_id}/{config_id}/load")
async def load_config_user(
    user_id: str,
    config_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Load a config from the weaviate database for a user.
    This will set the default config for all new trees associated with the user.
    It will also propagate the config to any trees set with default settings.
    It will not affect trees that have been set up with their own config.

    It will also update the ClientManager to use the new API keys.

    Args:
        user_id (str): The user ID.
        config_id (str): The config ID you want to load.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            config (dict): A dictionary of the updated config values.
    """
    logger.debug(f"/load_config_user API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {config_id}")

    warnings = []

    # Retrieve the user info
    try:
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        frontend_config: FrontendConfig = user["frontend_config"]
    except Exception as e:
        logger.exception(f"Error in /load_config_user API during user retrieval")
        return JSONResponse(
            content={
                "error": str(e),
                "config": {},
                "frontend_config": {},
                "warnings": warnings,
            }
        )

    try:

        # check if the user has a valid save location
        if (
            frontend_config.save_location_wcd_url == ""
            or frontend_config.save_location_wcd_api_key == ""
        ):
            raise Exception("WCD URL or API key not found.")

        if (
            frontend_config.save_location_wcd_url == ""
            or frontend_config.save_location_wcd_api_key == ""
        ):
            raise Exception(
                "No valid destination for config load location found. "
                "Please update the save location using the /update_save_location API."
            )

        # Retrieve the config from the weaviate database
        async with (
            frontend_config.save_location_client_manager.connect_to_async_client() as client
        ):
            uuid = generate_uuid5(config_id)
            collection = client.collections.get("ELYSIA_CONFIG__")
            config_response = await collection.query.fetch_object_by_id(uuid=uuid)
            if config_response is None:
                raise Exception(f"Config with ID {config_id} not found.")

            config_item: dict = config_response.properties  # type: ignore

        format_dict_to_serialisable(config_item)

        if "settings" in config_item:
            try:
                config_item["settings"] = decrypt_api_keys(config_item["settings"])
            except InvalidToken:
                warnings.append(
                    "Invalid token for config. You will need to re-enter your API keys in the settings."
                )
                config_item["settings"]["API_KEYS"] = {}
                config_item["settings"]["WCD_API_KEY"] = ""

        # Rename the keys to the correct format
        renamed_config = rename_keys(config_item)

        # Load the settings
        await user_manager.update_config(
            user_id,
            conversation_id=None,
            config_id=renamed_config["config_id"],
            config_name=renamed_config["name"],
            settings=renamed_config["settings"],
            style=renamed_config["style"],
            agent_description=renamed_config["agent_description"],
            end_goal=renamed_config["end_goal"],
            branch_initialisation=renamed_config["branch_initialisation"],
        )

        # Update the frontend config
        await user_manager.update_frontend_config(
            user_id, renamed_config["frontend_config"]
        )

    except Exception as e:
        tree_manager = user["tree_manager"]
        config = tree_manager.config
        frontend_config = user["frontend_config"]
        logger.exception(f"Error in /load_config API")
        return JSONResponse(
            content={
                "error": str(e),
                "config": config.to_json(),
                "frontend_config": frontend_config.to_json(),
                "warnings": warnings,
            }
        )

    tree_manager = user["tree_manager"]
    config = tree_manager.config
    frontend_config = user["frontend_config"]
    return JSONResponse(
        content={
            "error": "",
            "config": config.to_json(),
            "frontend_config": frontend_config.to_json(),
            "warnings": warnings,
        }
    )


@router.delete("/{user_id}/{config_id}")
async def delete_config(
    user_id: str,
    config_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Delete a config from the weaviate database.
    """
    logger.debug(f"/delete_config API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Config ID: {config_id}")

    try:
        user = await user_manager.get_user_local(user_id)
        frontend_config: FrontendConfig = user["frontend_config"]
        async with (
            frontend_config.save_location_client_manager.connect_to_async_client()
        ) as client:
            collection = client.collections.get("ELYSIA_CONFIG__")
            uuid = generate_uuid5(config_id)
            await collection.data.delete_by_id(uuid)

    except Exception as e:
        logger.exception(f"Error in /delete_config API")
        return JSONResponse(content={"error": str(e)})

    return JSONResponse(content={"error": ""})


@router.get("/{user_id}/list")
async def list_configs(
    user_id: str,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    List all the configs for a user.

    Args:
        data (ListConfigsData): A class with the following attributes:
            user_id (str): The user ID.
        user_manager (UserManager): The user manager instance.

    Returns:
        JSONResponse: A JSON response with the following attributes:
            error (str): An error message. Empty if no error.
            configs (list): A list of config IDs.
    """
    logger.debug(f"/list_configs API request received")
    logger.debug(f"User ID: {user_id}")

    headers = {"Cache-Control": "no-cache"}
    warnings = []

    try:
        user = await user_manager.get_user_local(user_id)

        if (
            "frontend_config" not in user
            or "save_location_wcd_url" not in user["frontend_config"].__dict__
            or "save_location_wcd_api_key" not in user["frontend_config"].__dict__
            or user["frontend_config"].save_location_wcd_url is None
            or user["frontend_config"].save_location_wcd_api_key is None
            or user["frontend_config"].save_location_wcd_url == ""
            or user["frontend_config"].save_location_wcd_api_key == ""
        ):
            logger.warning(
                "In /list_configs API, "
                "no valid destination for config location found. "
                "Returning no error but an empty list of configs."
            )
            warnings.append(
                "No valid destination for config location found. "
                "Cannot show saved configs."
            )
            return JSONResponse(
                content={"error": "", "configs": [], "warnings": warnings},
                headers=headers,
            )

        if not user["frontend_config"].save_location_client_manager.is_client:
            logger.warning(
                "In /list_configs API, "
                "Client manager is not connected to a client. "
                "Returning no error but an empty list of configs."
            )
            warnings.append(
                "Weaviate client is not connected. Cannot show saved configs."
            )
            return JSONResponse(
                content={"error": "", "configs": [], "warnings": warnings},
                headers=headers,
            )

        if user_id is not None:
            user_id_filter = Filter.by_property("user_id").equal(user_id)
        else:
            user_id_filter = None

        async with user[
            "frontend_config"
        ].save_location_client_manager.connect_to_async_client() as client:

            if await client.collections.exists("ELYSIA_CONFIG__"):
                collection = client.collections.get("ELYSIA_CONFIG__")
                len_collection = (
                    await collection.aggregate.over_all(total_count=True)
                ).total_count

                response = await collection.query.fetch_objects(
                    limit=len_collection,
                    sort=Sort.by_update_time(ascending=False),
                    return_metadata=MetadataQuery(last_update_time=True),
                    filters=user_id_filter,
                )

                configs = [
                    {
                        "config_id": obj.properties["config_id"],
                        "name": obj.properties["name"],
                        "default": obj.properties["default"],
                        "last_update_time": format_datetime(
                            obj.metadata.last_update_time
                        ),
                    }
                    for obj in response.objects
                ]
            else:
                configs = []

    except Exception as e:
        logger.exception(f"Error in /list_configs API")
        return JSONResponse(
            content={"error": str(e), "configs": [], "warnings": warnings},
            headers=headers,
        )

    return JSONResponse(
        content={"error": "", "configs": configs, "warnings": warnings}, headers=headers
    )
