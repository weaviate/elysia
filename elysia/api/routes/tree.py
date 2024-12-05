from fastapi import APIRouter, Depends
from elysia.api.api_types import InitialiseTreeData
from elysia.api.services.tree import TreeManager
from elysia.api.dependencies.common import get_tree_manager
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/initialise_tree")
async def initialise_tree(
    data: InitialiseTreeData,
    tree_manager: TreeManager = Depends(get_tree_manager)
):
    error = tree_manager.add_tree(data.user_id, data.conversation_id)
    return JSONResponse(
        content={
            "conversation_id": data.conversation_id,
            "tree": tree_manager.get_tree(data.user_id, data.conversation_id).tree, 
            "error": error
        }, 
        status_code=200
    )
