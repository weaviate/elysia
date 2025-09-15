from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from cryptography.fernet import InvalidToken

from elysia.api.api_types import InitialiseTreeData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.util.client import ClientManager
from elysia.api.utils.config import Config
from elysia.api.utils.encryption import decrypt_api_keys

from weaviate.classes.query import Filter

router = APIRouter()


async def get_default_config(
    client_manager: ClientManager,
    user_id: str,
    config_collection_name: str = "ELYSIA_CONFIG__",
) -> Config | None:
    async with client_manager.connect_to_async_client() as client:
        if not await client.collections.exists(config_collection_name):
            return None

        collection = client.collections.get(config_collection_name)
        default_configs = await collection.query.fetch_objects(
            filters=Filter.all_of(
                [
                    Filter.by_property("default").equal(True),
                    Filter.by_property("user_id").equal(user_id),
                ]
            )
        )

        if len(default_configs.objects) > 0:
            default_config: dict = default_configs.objects[0].properties  # type: ignore
            try:
                default_config["settings"] = decrypt_api_keys(
                    default_config["settings"]
                )
            except InvalidToken:
                logger.warning(f"Invalid token for default config, returning None")
                return None
            return Config.from_json(default_config)
        else:
            return None


@router.post("/user/{user_id}")
async def initialise_user(
    user_id: str, user_manager: UserManager = Depends(get_user_manager)
):
    """
    Create or retrieve a user.

    Returns:
        (JSONResponse): A JSON response with the following fields:
            - error (str): Any error message (empty string if no error).
            - user_exists (bool): True if the user exists, False otherwise.
            - config (dict): The user's config.
            - frontend_config (dict): The user's frontend config.
    """
    logger.debug(f"/initialise_user API request received")

    try:

        user_exists = user_manager.user_exists(user_id)

        # if a user does not exist, create a user and set up the configs
        if not user_exists:
            await user_manager.add_user_local(
                user_id,
            )  # leave config empty to create defaults for a new user

            # find any default configs
            if user_manager.users[user_id][
                "frontend_config"
            ].save_location_client_manager.is_client:
                default_config = await get_default_config(
                    user_manager.users[user_id][
                        "frontend_config"
                    ].save_location_client_manager,
                    user_id,
                )
                if default_config:
                    await user_manager.update_config(
                        user_id,
                        config_id=default_config.id,
                        config_name=default_config.name,
                        settings=default_config.settings.to_json(),
                        style=default_config.style,
                        agent_description=default_config.agent_description,
                        end_goal=default_config.end_goal,
                        branch_initialisation=default_config.branch_initialisation,
                    )
                    logger.debug("Using default config")

        # if a user exists, get the existing configs
        user = await user_manager.get_user_local(user_id)
        config = user["tree_manager"].config.to_json()
        frontend_config = user["frontend_config"].to_json()

        correct_settings = user["tree_manager"].config.settings.check()

        logger.debug(f"--------------------------------")
        logger.debug(f"Output of init/user:")
        logger.debug(f"User ID: {user_id}")
        logger.debug(f"User exists: {user_exists}")
        logger.debug(f"Config: {config}")
        logger.debug(f"Frontend config: {frontend_config}")
        logger.debug(f"Correct settings: {correct_settings}")

    except Exception as e:
        logger.exception(f"Error in /initialise_user API")
        return JSONResponse(
            content={
                "error": str(e),
                "user_exists": None,
                "config": {},
                "frontend_config": {},
                "correct_settings": {},
            },
            status_code=500,
        )
    return JSONResponse(
        content={
            "error": "",
            "user_exists": user_exists,
            "config": config,
            "frontend_config": frontend_config,
            "correct_settings": correct_settings,
        },
        status_code=200,
    )


@router.post("/tree/{user_id}/{conversation_id}")
async def initialise_tree(
    user_id: str,
    conversation_id: str,
    data: InitialiseTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    """
    Create or retrieve a tree.

    Returns:
        (JSONResponse): A JSON response with the following fields:
            - conversation_id (str): The conversation ID.
            - tree (dict): The tree.
            - error (str): Any error message (empty string if no error).
    """
    logger.debug(f"/initialise_tree API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Conversation ID: {conversation_id}")
    logger.debug(f"Low Memory: {data.low_memory}")

    try:
        tree = await user_manager.initialise_tree(
            user_id,
            conversation_id,
            low_memory=data.low_memory,
        )
    except Exception as e:
        logger.exception(f"Error in /initialise_tree API")
        return JSONResponse(
            content={
                "conversation_id": conversation_id,
                "tree": "",
                "error": str(e),
            },
            status_code=500,
        )

    return JSONResponse(
        content={
            "conversation_id": conversation_id,
            "tree": tree.tree,
            "error": "",
        },
        status_code=200,
    )
