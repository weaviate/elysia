import pytest
import os
from uuid import uuid4

from elysia.api.utils.config import default_presets
from weaviate.util import generate_uuid5
from elysia.api.services.user import UserManager
from elysia.api.api_types import (
    InitialiseTreeData,
    SaveConfigUserData,
    TreeNode,
    TreeGraph,
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
    add_tool_preset,
    get_tool_presets,
    delete_tool_preset,
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
    assert "tell_a_joke" in response["tools"]
    assert response["tools"]["tell_a_joke"]["description"] == joke_tool.description
    assert response["tools"]["tell_a_joke"]["inputs"].keys() == joke_tool.inputs.keys()
    assert response["tools"]["tell_a_joke"]["end"] == joke_tool.end
    assert response["tools"]["tell_a_joke"]["name"] == joke_tool.name


"""
Tests when not saving configs to Weaviate.
"""


test_presets = [
    TreeGraph(
        id="default",
        name="Default",
        default=True,
        nodes={
            "x12343": TreeNode(
                id="x12343",
                name="base",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a base-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=True,
            ),
            "y12321": TreeNode(id="y12321", name="query", is_branch=False),
            "z12321": TreeNode(id="z12321", name="aggregate", is_branch=False),
            "a12321": TreeNode(id="a12321", name="cited_summarize", is_branch=False),
            "b12321": TreeNode(id="b12321", name="text_response", is_branch=False),
        },
        edges=[
            ("x12343", "y12321"),
            ("x12343", "z12321"),
            ("x12343", "a12321"),
            ("x12343", "b12321"),
        ],
    ),
    TreeGraph(
        id="edward_preset_1",
        name="Edward Preset 1",
        default=False,
        nodes={
            "base": TreeNode(
                id="base",
                name="base",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a base-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=True,
            ),
            "epic_secondary_branch": TreeNode(
                id="epic_secondary_branch",
                name="epic_secondary_branch",
                is_branch=True,
                description="Lil buddy this branch is sick yo",
                instruction="Pick between doing a big query or a little aggregate bro.",
                is_root=False,
            ),
            "query": TreeNode(id="query", name="query", is_branch=False),
            "aggregate": TreeNode(id="aggregate", name="aggregate", is_branch=False),
            "cited_summarize": TreeNode(
                id="cited_summarize", name="cited_summarize", is_branch=False
            ),
            "text_response": TreeNode(
                id="text_response", name="text_response", is_branch=False
            ),
        },
        edges=[
            ("base", "epic_secondary_branch"),
            ("epic_secondary_branch", "query"),
            ("epic_secondary_branch", "aggregate"),
            ("epic_secondary_branch", "cited_summarize"),
            ("query", "cited_summarize"),
            ("epic_secondary_branch", "text_response"),
            ("query", "text_response"),
            ("cited_summarize", "text_response"),
        ],
    ),
    TreeGraph(
        id="edward_preset_2",
        name="Edward Preset 2",
        default=False,
        nodes={
            "this_is_the_root_branch": TreeNode(
                id="this_is_the_root_branch",
                name="this_is_the_root_branch",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a base-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=True,
            ),
            "this_is_the_second_branch": TreeNode(
                id="this_is_the_second_branch",
                name="this_is_the_second_branch",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a secondary-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=False,
            ),
            "this_is_the_third_branch": TreeNode(
                id="this_is_the_third_branch",
                name="this_is_the_third_branch",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a third-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=False,
            ),
            "this_is_the_fourth_branch": TreeNode(
                id="this_is_the_fourth_branch",
                name="this_is_the_fourth_branch",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a fourth-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=False,
            ),
            "this_is_the_fifth_branch": TreeNode(
                id="this_is_the_fifth_branch",
                name="this_is_the_fifth_branch",
                is_branch=True,
                description="",
                instruction=(
                    "Choose a fifth-level task based on the user's prompt and available information. "
                    "Decide based on the tools you have available as well as their descriptions. "
                    "Read them thoroughly and match the actions to the user prompt."
                ),
                is_root=False,
            ),
            "tell_a_joke": TreeNode(
                id="tell_a_joke", name="tell_a_joke", is_branch=False
            ),
            "basic_linear_regression_tool": TreeNode(
                id="basic_linear_regression_tool",
                name="basic_linear_regression_tool",
                is_branch=False,
            ),
            "text_response": TreeNode(
                id="text_response", name="text_response", is_branch=False
            ),
            "query": TreeNode(id="query", name="query", is_branch=False),
            "cited_summarize": TreeNode(
                id="cited_summarize", name="cited_summarize", is_branch=False
            ),
        },
        edges=[
            ("this_is_the_root_branch", "this_is_the_second_branch"),
            ("this_is_the_second_branch", "this_is_the_third_branch"),
            ("this_is_the_third_branch", "this_is_the_fourth_branch"),
            ("this_is_the_fourth_branch", "this_is_the_fifth_branch"),
            ("this_is_the_second_branch", "tell_a_joke"),
            ("this_is_the_third_branch", "basic_linear_regression_tool"),
            ("this_is_the_fourth_branch", "text_response"),
            ("this_is_the_fifth_branch", "query"),
            ("this_is_the_fifth_branch", "cited_summarize"),
            ("query", "cited_summarize"),
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("test_preset", test_presets)
async def test_cycle(test_preset: TreeGraph):
    user_id = f"test_add_tool_preset_{uuid4()}"
    user_manager = get_user_manager()
    user = await initialise_user(user_id, user_manager)

    # save config so that save_configs_to_weaviate is False
    response = await save_config_user(
        user_id=user_id,
        config_id=f"test_add_tool_preset_{uuid4()}",
        data=SaveConfigUserData(
            name="test_add_tool_preset",
            default=True,
            config={},
            frontend_config={"save_configs_to_weaviate": False},
        ),
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    user_local = await user_manager.get_user_local(user_id)
    tree_graph_manager = user_local["tree_graph_manager"]
    for preset in tree_graph_manager.presets:
        tree_graph_manager.remove(preset.id)

    # get available tools
    response = await get_available_tools()
    available_tools = read_response(response)["tools"]

    # add tool preset
    response = await add_tool_preset(
        user_id=user_id,
        data=test_preset,
        user_manager=user_manager,
    )
    response = read_response(response)
    assert response["error"] == ""

    user_local = await user_manager.get_user_local(user_id)
    tree_graph_manager = user_local["tree_graph_manager"]
    assert len(tree_graph_manager.presets) == 1
    assert tree_graph_manager.presets[-1].id == test_preset.id
    assert tree_graph_manager.presets[-1].name == test_preset.name

    # get tool presets
    response = await get_tool_presets(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == ""
    assert "presets" in response
    assert response["presets"] is not None
    assert len(response["presets"]) == 1
    assert response["presets"][-1]["id"] == test_preset.id
    assert response["presets"][-1]["name"] == test_preset.name

    # remove tool preset
    response = await delete_tool_preset(
        user_id=user_id, preset_id=test_preset.id, user_manager=user_manager
    )
    response = read_response(response)
    assert response["error"] == ""

    # check if its been deleted
    response = await get_tool_presets(user_id=user_id, user_manager=user_manager)
    response = read_response(response)
    assert response["error"] == ""
    assert "presets" in response
    assert response["presets"] is not None
    assert len(response["presets"]) == 0
