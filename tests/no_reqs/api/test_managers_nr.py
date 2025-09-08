import pytest
import os
import dotenv

from elysia.api.services.user import UserManager
from elysia.api.services.tree import TreeManager
from elysia.config import Settings
from elysia.api.utils.config import Config

dotenv.load_dotenv(override=True)

from uuid import uuid4


@pytest.mark.asyncio
async def test_user_manager():
    """
    Test basic user manager functionality.
    """
    user_id = f"test_{uuid4()}"
    conversation_id = f"test_{uuid4()}"

    user_manager = UserManager()

    # will raise error if no user created
    with pytest.raises(ValueError):
        await user_manager.get_user_local(user_id)

    # add a user
    await user_manager.add_user_local(user_id)

    # get user
    user = await user_manager.get_user_local(user_id)
    assert user is not None

    # will raise error if no tree created
    with pytest.raises(ValueError):
        await user_manager.get_tree(user_id, conversation_id)

    # add a tree
    await user_manager.initialise_tree(user_id, conversation_id, low_memory=True)

    # get tree
    tree = await user_manager.get_tree(user_id, conversation_id)
    assert tree is not None


@pytest.mark.asyncio
async def test_tree_manager():
    """
    Test basic tree manager functionality.
    """
    user_id = f"test_{uuid4()}"
    conversation_id = f"test_{uuid4()}"

    tree_manager = TreeManager(user_id)

    # will raise error if no tree created
    with pytest.raises(ValueError):
        tree_manager.get_tree(conversation_id)

    # add a tree
    tree_manager.add_tree(conversation_id, low_memory=True)

    # get tree
    tree = tree_manager.get_tree(conversation_id)
    assert tree is not None
