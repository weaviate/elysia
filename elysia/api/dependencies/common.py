from elysia.api.services.tree import TreeManager
from elysia.api.core.config import settings

# Initialize the global tree_manager
tree_manager = TreeManager(settings.COLLECTION_NAMES)

def get_tree_manager() -> TreeManager:
    return tree_manager