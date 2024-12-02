import unittest

import asyncio

from elysia.api.app import set_collections, initialise_tree, tree_manager
from elysia.api.api_types import InitialiseTreeData, SetCollectionsData

class TestCollections(unittest.TestCase):

    user_id = "test_user"
    conversation_id = "test_conversation"
    collection_name = "ecommerce"

    def test_remove_collection_basic(self):

        # create a new tree
        data = InitialiseTreeData(user_id=self.user_id, conversation_id=self.conversation_id)
        asyncio.run(initialise_tree(data))

        # save the tree
        tree = tree_manager.get_tree(self.user_id, self.conversation_id)

        # view available collections
        action_data_collections = list(tree.action_data.collection_information.keys())
        querier_collections = tree.querier.collection_names
        aggregator_collections = tree.aggregator.collection_names

        # remove the collection
        new_collections = [collection for collection in action_data_collections if collection != self.collection_name]
        remove_collections_data = SetCollectionsData(
            conversation_id=self.conversation_id,
            user_id=self.user_id,
            collection_names=new_collections,
            remove_data=False
        )
        asyncio.run(set_collections(remove_collections_data))
        
        # view available collections
        action_data_collections = list(tree.action_data.collection_information.keys())
        querier_collections = tree.querier.collection_names
        aggregator_collections = tree.aggregator.collection_names
        
        self.assertFalse(self.collection_name in action_data_collections)
        self.assertFalse(self.collection_name in querier_collections)
        self.assertFalse(self.collection_name in aggregator_collections)

    def _find_object_in_collection(self, tree, collection_name):
        found = False
        for object in tree.decision_data.available_information.retrieved:
            if object.metadata["collection_name"] == collection_name:
                found = True
                break
        return found

    def test_remove_collection_remove_data(self):

        user_id = self.user_id
        conversation_id = self.conversation_id + "_2"
        collection_name = self.collection_name

        # create a new tree
        data = InitialiseTreeData(user_id=user_id, conversation_id=conversation_id)
        asyncio.run(initialise_tree(data))

        # save the tree
        tree = tree_manager.get_tree(user_id, conversation_id)
        tree.training_route = ["search", "query"]

        # view available collections
        action_data_collections = list(tree.action_data.collection_information.keys())
        querier_collections = tree.querier.collection_names
        aggregator_collections = tree.aggregator.collection_names

        # run the tree to get a sample object from ecommerce
        tree.process_sync("Retrieve a random object from the ecommerce collection.")

        # This should have added an object to the ecommerce collection
        self.assertTrue(self._find_object_in_collection(tree, collection_name))

        # remove the collection
        new_collections = [collection for collection in action_data_collections if collection != collection_name]
        remove_collections_data = SetCollectionsData(
            conversation_id=conversation_id,
            user_id=user_id,
            collection_names=new_collections,
            remove_data=True
        )
        asyncio.run(set_collections(remove_collections_data))

        # The object should no longer be in the collection
        self.assertFalse(self._find_object_in_collection(tree, collection_name))

if __name__ == "__main__":
    unittest.main()