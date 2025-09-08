import pytest
import os

from weaviate.util import generate_uuid5
from elysia.api.services.user import UserManager
from elysia.api.api_types import (
    InitialiseTreeData,
    AddToolToTreeData,
    RemoveToolFromTreeData,
    AddBranchToTreeData,
    RemoveBranchFromTreeData,
    SaveConfigUserData,
    UpdateFrontendConfigData,
)
from elysia.api.routes.query import process
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import (
    update_frontend_config,
    save_config_user,
    get_current_user_config,
)
from elysia.api.routes.tools import (
    get_available_tools,
    add_tool_to_tree,
    remove_tool_from_tree,
    add_branch_to_tree,
    remove_branch_from_tree,
)
from elysia.api.custom_tools import TellAJoke
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.routes.user_config import (
    update_frontend_config,
    save_config_user,
    get_current_user_config,
)
from elysia.api.routes.query import process

from fastapi.responses import JSONResponse
import json


def read_response(response: JSONResponse):
    return json.loads(response.body)


from elysia.api.core.log import logger, set_log_level

set_log_level("CRITICAL")


def read_response(response: JSONResponse):
    return json.loads(response.body)


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


@pytest.mark.asyncio
async def test_tools_exist():
    joke_tool = TellAJoke()
    response = await get_available_tools()
    response = read_response(response)
    assert response["error"] == ""
    assert "tools" in response
    assert response["tools"] is not None
    assert "TellAJoke" in response["tools"]
    assert response["tools"]["TellAJoke"]["description"] == joke_tool.description
    assert response["tools"]["TellAJoke"]["inputs"].keys() == joke_tool.inputs.keys()
    assert response["tools"]["TellAJoke"]["end"] == joke_tool.end
    assert response["tools"]["TellAJoke"]["name"] == joke_tool.name


@pytest.mark.asyncio
async def test_basic_add_tool_to_tree():
    user_id = "test_user_basic_add_tool_to_tree"
    conversation_id = "test_conversation_basic_add_tool_to_tree"

    user_manager = get_user_manager()
    await initialise_user_and_tree(user_id, conversation_id)

    tool = TellAJoke()

    # add tool to tree
    response = await add_tool_to_tree(
        user_id=user_id,
        data=AddToolToTreeData(
            tool_name="TellAJoke",
            branch_id="base",
            from_tool_ids=[],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # check returned tree is correct
    assert tool.name in response["tree"]["options"]
    assert response["tree"]["options"][tool.name]["description"] == tool.description
    assert response["tree"]["options"][tool.name]["id"] == tool.name
    assert response["tree"]["options"][tool.name]["branch"] is False

    # check tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tool.name in tree.tree["options"]
    assert tree.tree["options"][tool.name]["description"] == tool.description
    assert tree.tree["options"][tool.name]["id"] == tool.name
    assert tree.tree["options"][tool.name]["branch"] is False


@pytest.mark.asyncio
async def test_add_tool_to_tree_with_from_tool_ids():
    user_id = "test_user_add_tool_to_tree_with_from_tool_ids"
    conversation_id = "test_conversation_add_tool_to_tree_with_from_tool_ids"

    user_manager = get_user_manager()

    tool = TellAJoke()

    await initialise_user_and_tree(user_id, conversation_id)

    # add tool to tree
    response = await add_tool_to_tree(
        user_id=user_id,
        data=AddToolToTreeData(
            tool_name="TellAJoke",
            branch_id="base",
            from_tool_ids=["query"],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # check returned tree is correct
    assert tool.name in response["tree"]["options"]["query"]["options"]
    assert (
        response["tree"]["options"]["query"]["options"][tool.name]["description"]
        == tool.description
    )
    assert response["tree"]["options"]["query"]["options"][tool.name]["id"] == tool.name
    assert response["tree"]["options"]["query"]["options"][tool.name]["branch"] is False

    # check tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tool.name in tree.tree["options"]["query"]["options"]
    assert (
        tree.tree["options"]["query"]["options"][tool.name]["description"]
        == tool.description
    )
    assert tree.tree["options"]["query"]["options"][tool.name]["id"] == tool.name
    assert tree.tree["options"]["query"]["options"][tool.name]["branch"] is False


@pytest.mark.asyncio
async def test_basic_remove_tool_from_tree():
    user_id = "test_user_basic_remove_tool_from_tree"
    conversation_id = "test_conversation_basic_remove_tool_from_tree"

    user_manager = get_user_manager()

    tool = TellAJoke()

    await initialise_user_and_tree(user_id, conversation_id)

    # first add tool to tree
    response = await add_tool_to_tree(
        user_id=user_id,
        data=AddToolToTreeData(
            tool_name="TellAJoke",
            branch_id="base",
            from_tool_ids=[],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # remove tool from tree
    response = await remove_tool_from_tree(
        user_id=user_id,
        data=RemoveToolFromTreeData(
            tool_name="TellAJoke",
            branch_id="base",
            from_tool_ids=[],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # check returned tree is correct
    assert tool.name not in response["tree"]["options"]

    # check tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tool.name not in tree.tree["options"]


@pytest.mark.asyncio
async def test_remove_tool_from_tree_with_from_tool_ids():
    user_id = "test_user_remove_tool_from_tree_with_from_tool_ids"
    conversation_id = "test_conversation_remove_tool_from_tree_with_from_tool_ids"

    user_manager = get_user_manager()

    tool = TellAJoke()

    await initialise_user_and_tree(user_id, conversation_id)

    # first add tool to tree
    response = await add_tool_to_tree(
        user_id=user_id,
        data=AddToolToTreeData(
            tool_name="TellAJoke",
            branch_id="base",
            from_tool_ids=["query"],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # remove tool from tree
    response = await remove_tool_from_tree(
        user_id=user_id,
        data=RemoveToolFromTreeData(
            tool_name="TellAJoke",
            branch_id="base",
            from_tool_ids=["query"],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # check returned tree is correct
    assert tool.name not in response["tree"]["options"]["query"]["options"]

    # check tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tool.name not in tree.tree["options"]["query"]["options"]


@pytest.mark.asyncio
async def test_basic_add_branch_to_tree():
    user_id = "test_user_basic_add_branch_to_tree"
    conversation_id = "test_conversation_basic_add_branch_to_tree"

    user_manager = get_user_manager()

    await initialise_user_and_tree(user_id, conversation_id)

    # add branch to tree
    response = await add_branch_to_tree(
        user_id=user_id,
        data=AddBranchToTreeData(
            id="new_branch",
            description="New Branch Description",
            instruction="New Branch Instruction",
            from_branch_id="base",
            from_tool_ids=[],
            status="New Branch Status",
            root=False,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # check returned tree is correct
    assert "new_branch" in response["tree"]["options"]
    assert response["tree"]["options"]["new_branch"]["id"] == "new_branch"
    assert (
        response["tree"]["options"]["new_branch"]["description"]
        == "New Branch Description"
    )
    assert (
        response["tree"]["options"]["new_branch"]["instruction"]
        == "New Branch Instruction"
    )
    assert response["tree"]["options"]["new_branch"]["branch"] is True
    assert response["tree"]["options"]["new_branch"]["options"] == {}

    # check tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert "new_branch" in tree.tree["options"]
    assert tree.tree["options"]["new_branch"]["id"] == "new_branch"
    assert tree.tree["options"]["new_branch"]["description"] == "New Branch Description"
    assert tree.tree["options"]["new_branch"]["instruction"] == "New Branch Instruction"
    assert tree.tree["options"]["new_branch"]["branch"] is True
    assert tree.tree["options"]["new_branch"]["options"] == {}


@pytest.mark.asyncio
async def test_add_branch_to_branch():
    user_id = "test_user_add_branch_to_branch"
    conversation_id = "test_conversation_add_branch_to_branch"
    user_manager = get_user_manager()

    await initialise_user_and_tree(user_id, conversation_id)

    # get config_id
    response = await get_current_user_config(
        user_id,
        user_manager,
    )
    assert read_response(response)["error"] == ""
    config_id = read_response(response)["config"]["id"]
    config_name = read_response(response)["config"]["name"]

    # change branch initialisation
    response = await save_config_user(
        user_id,
        config_id,
        SaveConfigUserData(
            name=config_name,
            config={
                "branch_initialisation": "multi_branch",
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
            default=True,
        ),
        user_manager,
    )
    assert read_response(response)["error"] == ""

    # add branch to tree
    response = await add_branch_to_tree(
        user_id=user_id,
        data=AddBranchToTreeData(
            id="new_branch",
            description="New Branch Description",
            instruction="New Branch Instruction",
            from_branch_id="search",
            status="New Branch Status",
            from_tool_ids=[],
            root=False,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""
    assert response["tree"] is not None
    assert response["tree"] is not {}

    # check returned tree is correct
    assert "new_branch" in response["tree"]["options"]["search"]["options"]
    assert (
        response["tree"]["options"]["search"]["options"]["new_branch"]["id"]
        == "new_branch"
    )
    assert (
        response["tree"]["options"]["search"]["options"]["new_branch"]["description"]
        == "New Branch Description"
    )
    assert (
        response["tree"]["options"]["search"]["options"]["new_branch"]["instruction"]
        == "New Branch Instruction"
    )
    assert (
        response["tree"]["options"]["search"]["options"]["new_branch"]["branch"] is True
    )
    assert (
        response["tree"]["options"]["search"]["options"]["new_branch"]["options"] == {}
    )

    # check tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert "new_branch" in tree.tree["options"]["search"]["options"]
    assert tree.tree["options"]["search"]["options"]["new_branch"]["id"] == "new_branch"
    assert (
        tree.tree["options"]["search"]["options"]["new_branch"]["description"]
        == "New Branch Description"
    )
    assert (
        tree.tree["options"]["search"]["options"]["new_branch"]["instruction"]
        == "New Branch Instruction"
    )
    assert tree.tree["options"]["search"]["options"]["new_branch"]["branch"] is True
    assert tree.tree["options"]["search"]["options"]["new_branch"]["options"] == {}


@pytest.mark.asyncio
async def test_full_cycle():
    user_id = "test_user_full_cycle"
    conversation_id = "test_conversation_full_cycle"

    user_manager = get_user_manager()

    await initialise_user_and_tree(user_id, conversation_id)

    # get config_id
    response = await get_current_user_config(
        user_id,
        user_manager,
    )
    assert read_response(response)["error"] == ""
    config_id = read_response(response)["config"]["id"]
    config_name = read_response(response)["config"]["name"]

    # change branch initialisation
    response = await save_config_user(
        user_id,
        config_id,
        SaveConfigUserData(
            name=config_name,
            config={
                "branch_initialisation": "multi_branch",
            },
            frontend_config={
                "save_configs_to_weaviate": False,
            },
            default=True,
        ),
        user_manager,
    )
    assert read_response(response)["error"] == ""

    # add a branch to a tool
    response = await add_branch_to_tree(
        user_id=user_id,
        data=AddBranchToTreeData(
            id="new_branch",
            description="New Branch Description",
            instruction="New Branch Instruction",
            from_branch_id="search",
            from_tool_ids=["query"],
            status="New Branch Status",
            root=False,
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # check the tree is correct
    assert response["tree"] is not None
    assert response["tree"] != {}
    assert (
        "new_branch"
        in response["tree"]["options"]["search"]["options"]["query"]["options"]
    )
    assert (
        response["tree"]["options"]["search"]["options"]["query"]["options"][
            "new_branch"
        ]["id"]
        == "new_branch"
    )

    # get tree and check there is the new branch
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert "new_branch" in tree.tree["options"]["search"]["options"]["query"]["options"]
    assert (
        tree.tree["options"]["search"]["options"]["query"]["options"]["new_branch"][
            "id"
        ]
        == "new_branch"
    )

    # check decision nodes updated
    assert "new_branch" in tree.decision_nodes
    assert "search.query" in tree.decision_nodes
    assert "new_branch" in tree.decision_nodes["search.query"].options

    # add a tool to the new branch
    response = await add_tool_to_tree(
        user_id=user_id,
        data=AddToolToTreeData(
            tool_name="TellAJoke",
            branch_id="new_branch",
            from_tool_ids=[],
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # check the tree is correct
    assert (
        "tell_a_joke"
        in response["tree"]["options"]["search"]["options"]["query"]["options"][
            "new_branch"
        ]["options"]
    )

    # and tree itself is correct
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert (
        "tell_a_joke"
        in tree.tree["options"]["search"]["options"]["query"]["options"]["new_branch"][
            "options"
        ]
    )

    # check decision nodes updated
    assert "tell_a_joke" in tree.decision_nodes["new_branch"].options

    # remove the branch from the tree
    response = await remove_branch_from_tree(
        user_id=user_id,
        data=RemoveBranchFromTreeData(id="new_branch"),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    # check the branch and its stemmed tools are gone
    assert (
        "new_branch"
        not in response["tree"]["options"]["search"]["options"]["query"]["options"]
    )

    tree = await user_manager.get_tree(user_id, conversation_id)
    assert (
        "new_branch"
        not in tree.tree["options"]["search"]["options"]["query"]["options"]
    )
    assert "new_branch" not in tree.decision_nodes
