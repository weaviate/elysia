# FastAPI
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.api_types import (
    DebugData,
    FollowUpSuggestionsData,
    NERData,
    TitleData,
    MigrateDataData,
)

from elysia.tree.tree import Tree
from elysia.util.client import ClientManager

# Logging
from elysia.api.core.log import logger

# Dependencies
from elysia.api.dependencies.common import get_user_manager

# Services
from elysia.api.services.user import UserManager

# util
from elysia.api.core.log import logger
from elysia.api.utils.collection_migration import (
    migrate_data_to_multi_tenancy,
    migrate_data_both_multi_tenancy,
    set_elysia_version,
    reset_collections,
)

router = APIRouter()


@router.post("/title")
async def title(data: TitleData, user_manager: UserManager = Depends(get_user_manager)):
    logger.debug(f"/title API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")
    logger.debug(f"Text: {data.text}")

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        logger.warning(
            f"(/title) Conversation {data.conversation_id} has timed out for user {data.user_id}"
        )
        return JSONResponse(
            content={"title": "", "error": "Conversation has timed out"},
            status_code=401,
        )

    try:
        tree: Tree = await user_manager.get_tree(data.user_id, data.conversation_id)
        title = await tree.create_conversation_title_async()
        return JSONResponse(
            content={"title": title, "error": ""},
            status_code=200,
        )

    except Exception as e:
        logger.exception(f"Error in /title API")
        return JSONResponse(
            content={"title": "", "error": str(e)},
            status_code=200,
        )


@router.post("/follow_up_suggestions")
async def follow_up_suggestions(
    data: FollowUpSuggestionsData, user_manager: UserManager = Depends(get_user_manager)
):
    logger.debug(f"/follow_up_suggestions API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")

    if user_manager.check_tree_timeout(data.user_id, data.conversation_id):
        logger.warning(
            f"(/follow_up_suggestions) Conversation {data.conversation_id} has timed out for user {data.user_id}"
        )
        return JSONResponse(
            content={"suggestions": [], "error": "Conversation has timed out"},
            status_code=408,
        )

    try:
        # wait for tree event to be completed
        local_user = await user_manager.get_user_local(data.user_id)
        event = local_user["tree_manager"].get_event(data.conversation_id)
        # await event.wait()

        # get tree from user_id, conversation_id
        tree: Tree = await user_manager.get_tree(data.user_id, data.conversation_id)

        suggestions = await tree.get_follow_up_suggestions_async()

        event.set()

        return JSONResponse(
            content={"suggestions": suggestions, "error": ""},
            status_code=200,
        )

    except Exception as e:
        logger.exception(f"Error in /follow_up_suggestions API")
        return JSONResponse(
            content={"suggestions": [], "error": str(e)}, status_code=200
        )


@router.post("/debug")
async def debug(data: DebugData, user_manager: UserManager = Depends(get_user_manager)):
    logger.debug(f"/debug API request received")
    logger.debug(f"User ID: {data.user_id}")
    logger.debug(f"Conversation ID: {data.conversation_id}")

    try:
        tree = await user_manager.get_tree(data.user_id, data.conversation_id)
        if tree.debug:
            base_lm = tree.base_lm
            complex_lm = tree.complex_lm

            histories = []
            for i, lm in enumerate([base_lm, complex_lm]):
                histories.append([])
                for lm_history in lm.history:
                    message_thread = []
                    for message in lm_history["messages"]:
                        if isinstance(message["content"], list):
                            # assume its a system message list with one dict element
                            message["content"] = message["content"][0]["text"]
                        message_thread.append(message)

                    message_thread.append(
                        {
                            "role": "assistant",
                            "content": lm_history["response"]
                            .choices[0]
                            .message.content,
                        }
                    )
                    histories[i].append(message_thread)

            out = {
                "base_lm": {"model": base_lm.model, "chat": histories[0]},
                "complex_lm": {"model": complex_lm.model, "chat": histories[1]},
            }
            return JSONResponse(content=out, status_code=200)

        else:
            return JSONResponse(content={}, status_code=200)
    except Exception as e:
        logger.exception(f"Error in /debug API")
        return JSONResponse(
            content={"base_lm": {}, "complex_lm": {}, "error": str(e)}, status_code=200
        )


@router.post("/migrate/{user_id}")
async def migrate(
    user_id: str,
    data: MigrateDataData,
    user_manager: UserManager = Depends(get_user_manager),
):
    logger.debug(f"/migrate API request received")
    logger.debug(f"User ID: {user_id}")
    logger.debug(f"Reset: {data.reset}")
    try:
        collection_names = [
            "ELYSIA_CONFIG__",
            "ELYSIA_TREES__",
            "ELYSIA_TOOL_PRESETS__",
            "ELYSIA_FEEDBACK__",
        ]
        user_local = await user_manager.get_user_local(user_id)
        save_location_client_manager: ClientManager = user_local[
            "frontend_config"
        ].save_location_client_manager
        async with save_location_client_manager.connect_to_async_client() as client:
            if data.reset:
                try:
                    await reset_collections(client)
                    logger.info(f"Reset collections")
                    return JSONResponse(content={"error": ""}, status_code=200)
                except Exception as e:
                    logger.exception(f"Error in resetting collections")
                    return JSONResponse(content={"error": str(e)}, status_code=500)

            try:
                for collection_name in collection_names:

                    # check if migration already in progress, delete temporary collections
                    if await client.collections.exists(f"{collection_name}_MIGRATED__"):
                        await client.collections.delete(f"{collection_name}_MIGRATED__")

                    # 0. check existence of collection
                    if not await client.collections.exists(collection_name):
                        continue

                    # 0.5 check existence of tenants in existing collection
                    collection = client.collections.get(collection_name)
                    try:
                        await collection.tenants.get()
                        multi_tenancy = True
                    except Exception as e:
                        multi_tenancy = False

                    if multi_tenancy:
                        continue

                    # 1. create a new collection and move data
                    await migrate_data_to_multi_tenancy(
                        client, collection_name, f"{collection_name}_MIGRATED__"
                    )

                    logger.info(f"Migrated data to {f"{collection_name}_MIGRATED__"}")

            except Exception as e:
                for collection_name in collection_names:
                    if await client.collections.exists(f"{collection_name}_MIGRATED__"):
                        await client.collections.delete(f"{collection_name}_MIGRATED__")

                logger.exception(
                    f"Error in migrating collections. Rolling back migrations."
                )
                return JSONResponse(
                    content={
                        "error": f"Error in migrating collections. Rolling back migrations."
                    },
                    status_code=500,
                )

            try:
                # do all migrations first before deleting and moving
                for collection_name in collection_names:

                    # 0. check existence of collection
                    if not await client.collections.exists(collection_name):
                        continue

                    # 0.5 check existence of tenants in existing collection
                    collection = client.collections.get(collection_name)
                    try:
                        await collection.tenants.get()
                        multi_tenancy = True
                    except Exception as e:
                        multi_tenancy = False

                    if multi_tenancy:
                        continue

                    # 2. delete old collection
                    await client.collections.delete(collection_name)
                    logger.info(f"Deleted old collection {collection_name}")

                    # 3. migrate data back to original named collection
                    await migrate_data_both_multi_tenancy(
                        client, f"{collection_name}_MIGRATED__", collection_name
                    )

                    logger.info(
                        f"Migrated data back to original named collection {collection_name}"
                    )

                    # 4. delete the temporary migrated collection
                    await client.collections.delete(f"{collection_name}_MIGRATED__")

                    logger.info(
                        f"Deleted temporary migrated collection {f"{collection_name}_MIGRATED__"}"
                    )

            except Exception as e:
                logger.exception(
                    f"Error in migrating collections during final migration. Some data may be lost."
                )
                # try to cleanup migrated collections
                for collection_name in collection_names:
                    if await client.collections.exists(f"{collection_name}_MIGRATED__"):
                        await client.collections.delete(f"{collection_name}_MIGRATED__")

                return JSONResponse(
                    content={
                        "error": f"Error in migrating collections during final migration. Some data may be lost."
                    },
                    status_code=500,
                )

            await set_elysia_version(client, 0.3)

    except Exception as e:
        logger.exception(f"Error in migrating collections")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    return JSONResponse(content={"error": ""}, status_code=200)
