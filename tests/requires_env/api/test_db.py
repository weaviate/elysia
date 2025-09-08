import pytest

from elysia.api.api_types import (
    InitialiseTreeData,
    QueryData,
    UpdateFrontendConfigData,
)
from elysia.api.routes.query import process
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import update_frontend_config
from elysia.api.routes.db import (
    save_tree,
    load_tree,
    get_saved_trees,
    delete_tree,
)

from fastapi.responses import JSONResponse
import json


def read_response(response: JSONResponse):
    return json.loads(response.body)


from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    results = []

    async def send_json(self, data: dict):
        self.results.append(data)


async def initialise_user_and_tree(user_id: str, conversation_id: str):
    user_manager = get_user_manager()

    response = await initialise_user(
        user_id,
        user_manager,
    )

    response = await initialise_tree(
        user_id,
        conversation_id,
        InitialiseTreeData(
            low_memory=False,
        ),
        user_manager,
    )


async def delete_user_and_tree(user_id: str, conversation_id: str):
    user_manager = get_user_manager()
    await user_manager.delete_tree(user_id, conversation_id)


class TestSaveLoad:

    @pytest.mark.asyncio
    async def test_manual_save_load_cycle(self):

        user_id = "test_user_full_save_load_cycle"
        conversation_id = "test_conversation_full_save_load_cycle"
        query_id = "test_query_full_save_load_cycle"

        websocket = fake_websocket()
        user_manager = get_user_manager()

        await initialise_user_and_tree(user_id, conversation_id)

        try:
            await update_frontend_config(
                user_id=user_id,
                data=UpdateFrontendConfigData(
                    config={
                        "save_trees_to_weaviate": False,
                    },
                ),
                user_manager=user_manager,
            )

            await process(
                QueryData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query="hi",
                    query_id=query_id,
                    collection_names=[],
                ).model_dump(),
                websocket,
                user_manager,
            )

            # check that the tree is not saved to weaviate (automatic saving turned off)
            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert conversation_id not in list(read_response(response)["trees"].keys())

            response = await save_tree(
                user_id=user_id,
                conversation_id=conversation_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""

            response = await load_tree(
                user_id=user_id,
                conversation_id=conversation_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert read_response(response)["rebuild"] != []
            for rebuilt in read_response(response)["rebuild"]:
                assert "type" in rebuilt
                assert "conversation_id" in rebuilt
                assert rebuilt["conversation_id"] == conversation_id
                assert "id" in rebuilt
                if "user_id" in rebuilt:
                    assert rebuilt["user_id"] == user_id

            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert len(list(read_response(response)["trees"].keys())) >= 1
            old_len = len(list(read_response(response)["trees"].keys()))
            assert "test_conversation_full_save_load_cycle" in list(
                read_response(response)["trees"].keys()
            )

            response = await delete_tree(
                user_id=user_id,
                conversation_id=conversation_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""

            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert len(list(read_response(response)["trees"].keys())) < old_len
            assert "test_conversation_full_save_load_cycle" not in list(
                read_response(response)["trees"].keys()
            )
        finally:
            await delete_user_and_tree(user_id, conversation_id)

    @pytest.mark.asyncio
    async def test_automatic_save_load_cycle(self):

        user_id = "test_user_automatic_save_load_cycle"
        conversation_id = "test_conversation_automatic_save_load_cycle"
        query_id = "test_query_automatic_save_load_cycle"

        websocket = fake_websocket()
        user_manager = get_user_manager()

        await initialise_user_and_tree(user_id, conversation_id)

        try:
            await update_frontend_config(
                user_id=user_id,
                data=UpdateFrontendConfigData(
                    config={
                        "save_trees_to_weaviate": True,
                    },
                ),
                user_manager=user_manager,
            )

            await process(
                QueryData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query="hi",
                    query_id=query_id,
                    collection_names=[],
                ).model_dump(),
                websocket,
                user_manager,
            )

            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert len(list(read_response(response)["trees"].keys())) >= 1
            old_len = len(list(read_response(response)["trees"].keys()))
            assert "test_conversation_automatic_save_load_cycle" in list(
                read_response(response)["trees"].keys()
            )

            response = await load_tree(
                user_id=user_id,
                conversation_id=conversation_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert read_response(response)["rebuild"] != []
            for rebuilt in read_response(response)["rebuild"]:
                assert "type" in rebuilt
                assert "conversation_id" in rebuilt
                assert rebuilt["conversation_id"] == conversation_id
                assert "id" in rebuilt
                if "user_id" in rebuilt:
                    assert rebuilt["user_id"] == user_id

            response = await delete_tree(
                user_id=user_id,
                conversation_id=conversation_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""

            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert len(list(read_response(response)["trees"].keys())) < old_len
            assert "test_conversation_automatic_save_load_cycle" not in list(
                read_response(response)["trees"].keys()
            )
        finally:
            await delete_user_and_tree(user_id, conversation_id)

    @pytest.mark.asyncio
    async def test_automatic_load_on_tree_timeout(self):

        user_id = "test_user_automatic_load_on_tree_timeout"
        conversation_id = "test_conversation_automatic_load_on_tree_timeout"
        query_id = "test_query_automatic_load_on_tree_timeout"

        websocket = fake_websocket()
        user_manager = get_user_manager()

        await initialise_user_and_tree(user_id, conversation_id)

        try:

            # do a query
            await process(
                QueryData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query="hi",
                    query_id=query_id,
                    collection_names=[],
                ).model_dump(),
                websocket,
                user_manager,
            )

            # store number of outputs in websocket results
            num_outputs = len(websocket.results)

            # the tree should be saved to weaviate
            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert len(list(read_response(response)["trees"].keys())) >= 1

            # timeout the tree manually by deleting it from the treemanager
            local_user = await user_manager.get_user_local(user_id)
            tree_manager = local_user["tree_manager"]
            del tree_manager.trees[conversation_id]

            # check the tree exists in weaviate
            tree_exists_response = await user_manager.check_tree_exists_weaviate(
                user_id, conversation_id
            )
            assert tree_exists_response

            # do a query - this should not error
            await process(
                QueryData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query="hi again",
                    query_id=query_id,
                    collection_names=[],
                ).model_dump(),
                websocket,
                user_manager,
            )
            assert len(websocket.results) > num_outputs
            assert "type" in websocket.results[-1]
            assert websocket.results[-1]["type"] != "tree_timeout_error"
        finally:
            await delete_user_and_tree(user_id, conversation_id)

    @pytest.mark.asyncio
    async def test_saved_trees_with_different_users(self):

        user_id = "test_user_saved_trees_with_different_users_1"
        conversation_id = "test_conversation_saved_trees_with_different_users_1"
        query_id = "test_query_saved_trees_with_different_users_1"

        user_id_2 = "test_user_saved_trees_with_different_users_2"
        conversation_id_2 = "test_conversation_saved_trees_with_different_users_2"
        query_id_2 = "test_query_saved_trees_with_different_users_2"

        websocket = fake_websocket()
        user_manager = get_user_manager()

        await initialise_user_and_tree(user_id, conversation_id)
        await initialise_user_and_tree(user_id_2, conversation_id_2)
        try:

            # do a query with this user, automatically saves
            await process(
                QueryData(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    query="hi",
                    query_id=query_id,
                    collection_names=[],
                ).model_dump(),
                websocket,
                user_manager,
            )

            # do a query with this user, automatically saves
            await process(
                QueryData(
                    user_id=user_id_2,
                    conversation_id=conversation_id_2,
                    query="hi",
                    query_id=query_id_2,
                    collection_names=[],
                ).model_dump(),
                websocket,
                user_manager,
            )

            # check that the trees exist in two separate places
            response = await get_saved_trees(
                user_id=user_id,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert conversation_id in list(read_response(response)["trees"].keys())
            assert conversation_id_2 not in list(
                read_response(response)["trees"].keys()
            )

            # check that the trees exist in two separate places
            response = await get_saved_trees(
                user_id=user_id_2,
                user_manager=user_manager,
            )
            assert read_response(response)["error"] == ""
            assert conversation_id not in list(read_response(response)["trees"].keys())
            assert conversation_id_2 in list(read_response(response)["trees"].keys())
        finally:
            await delete_user_and_tree(user_id, conversation_id)
            await delete_user_and_tree(user_id_2, conversation_id_2)
