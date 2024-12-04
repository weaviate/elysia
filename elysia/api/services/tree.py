from elysia.tree.tree import Tree

class TreeManager:
    """
    Manages trees for different conversations.
    """

    def __init__(
            self, 
            collection_names: list[str]
        ):
        self.trees = {}
        self.collection_names = collection_names

    def add_tree(self, user_id: str, conversation_id: str):
        
        if user_id not in self.trees:
            self.trees[user_id] = {}
        
        if conversation_id not in self.trees[user_id]:
            self.trees[user_id][conversation_id] = Tree(verbosity=2, conversation_id=conversation_id, collection_names=self.collection_names)
        
        return self.trees[user_id][conversation_id].initialise_error_message

    def get_tree(self, user_id: str, conversation_id: str):
        if user_id not in self.trees :
            self.add_tree(user_id, conversation_id)
        elif conversation_id not in self.trees[user_id]:
            self.add_tree(user_id, conversation_id)
        return self.trees[user_id][conversation_id]