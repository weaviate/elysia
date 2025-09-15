import elysia.api.custom_tools as custom_tools
from elysia import Tool
from typing import Dict, Type
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.core.log import logger
from elysia.tree.tree import Tree
from elysia.api.api_types import (
    AddToolToTreeData,
    RemoveToolFromTreeData,
    AddBranchToTreeData,
    RemoveBranchFromTreeData,
)
from elysia.api.services.tree import TreeManager

router = APIRouter()


def find_tool_classes() -> Dict[str, Type[Tool]]:
    """
    Find all Tool subclasses in the custom_tools module.

    Returns:
        (dict): Dictionary mapping class names to Tool subclass types
    """
    tool_classes = {}

    # Get all objects from the custom_tools module
    module_objects = dict(
        [
            (name, cls)
            for name, cls in custom_tools.__dict__.items()
            if isinstance(cls, type)
        ]
    )

    # Filter for Tool subclasses (excluding the base Tool class)
    for name, cls in module_objects.items():
        if issubclass(cls, Tool) and cls.__name__ != "Tool":
            tool_classes[name] = cls

    return tool_classes


def find_tool_metadata() -> Dict[str, Dict[str, str]]:
    """
    Find all tool metadata in the custom_tools module.

    Returns:
        (dict): Dictionary mapping tool names to tool metadata
    """
    tools = find_tool_classes()
    metadata = {}
    for name, cls in tools.items():
        metadata[name] = cls.get_metadata()
    return metadata


@router.get("/available")
async def get_available_tools():
    headers = {"Cache-Control": "no-cache"}

    try:
        return JSONResponse(
            content={"tools": find_tool_metadata(), "error": ""},
            status_code=200,
            headers=headers,
        )
    except Exception as e:
        logger.error(f"Error getting available tools: {str(e)}")
        return JSONResponse(
            content={"tools": {}, "error": str(e)},
            status_code=500,
            headers=headers,
        )


@router.post("/{user_id}/add")
async def add_tool_to_tree(
    user_id: str,
    data: AddToolToTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        # get tool class
        tool_class = find_tool_classes()[data.tool_name]
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]

        # get tree and add tool
        for conversation_id in tree_manager.trees:
            tree: Tree = tree_manager.get_tree(conversation_id)
            tree.add_tool(tool_class, data.branch_id, from_tool_ids=data.from_tool_ids)

        # all trees should look the same, so return the most recent one
        return JSONResponse(content={"tree": tree.tree, "error": ""}, status_code=200)

    except Exception as e:
        logger.error(f"Error adding tool to tree: {str(e)}")
        return JSONResponse(content={"tree": {}, "error": str(e)}, status_code=500)


@router.post("/{user_id}/remove")
async def remove_tool_from_tree(
    user_id: str,
    data: RemoveToolFromTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        # get tool class
        tool_class = find_tool_classes()[data.tool_name]

        # get tree and remove tool
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        for conversation_id in tree_manager.trees:
            tree: Tree = tree_manager.get_tree(conversation_id)
            tree.remove_tool(
                tool_class.get_metadata()["name"],  # type: ignore
                data.branch_id,
                from_tool_ids=data.from_tool_ids,
            )

        return JSONResponse(content={"tree": tree.tree, "error": ""}, status_code=200)
    except Exception as e:
        logger.error(f"Error removing tool from tree: {str(e)}")
        return JSONResponse(content={"tree": {}, "error": str(e)}, status_code=500)


@router.post("/{user_id}/add_branch")
async def add_branch_to_tree(
    user_id: str,
    data: AddBranchToTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        # get tree and add branch
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        for conversation_id in tree_manager.trees:
            tree: Tree = tree_manager.get_tree(conversation_id)
            tree.add_branch(
                branch_id=data.id,
                instruction=data.instruction,
                description=data.description,
                root=data.root,
                from_branch_id=data.from_branch_id,
                from_tool_ids=data.from_tool_ids,
                status=data.status,
            )

        return JSONResponse(content={"tree": tree.tree, "error": ""}, status_code=200)
    except Exception as e:
        logger.error(f"Error adding branch to tree: {str(e)}")
        return JSONResponse(content={"tree": {}, "error": str(e)}, status_code=500)


@router.post("/{user_id}/remove_branch")
async def remove_branch_from_tree(
    user_id: str,
    data: RemoveBranchFromTreeData,
    user_manager: UserManager = Depends(get_user_manager),
):
    try:
        # get tree and remove branch
        user = await user_manager.get_user_local(user_id)
        tree_manager: TreeManager = user["tree_manager"]
        for conversation_id in tree_manager.trees:
            tree: Tree = tree_manager.get_tree(conversation_id)
            tree.remove_branch(data.id)

        return JSONResponse(content={"tree": tree.tree, "error": ""}, status_code=200)
    except Exception as e:
        logger.error(f"Error removing branch from tree: {str(e)}")
        return JSONResponse(content={"tree": {}, "error": str(e)}, status_code=500)
