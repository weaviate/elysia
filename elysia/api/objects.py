import uuid

class Branch:
    """
    These branches are searched in advance of the tree being run, iterating through code and function calls.
    Used to build the tree.
    """
    def __init__(self, updates: dict):
        self.updates = updates

class Update:
    """
    Base class for all updates.
    """
    def __init__(self, type: str):
        self.type = type

    def to_frontend(self, conversation_id: str, query_id: str, payload: dict):
        return {
            "type": self.type,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "id": self.type[:3] + "-" + str(uuid.uuid4()),
            "payload": payload
        }
    
    @staticmethod
    def to_frontend_json(type: str, conversation_id: str, query_id: str, payload: dict):
        return {
            "type": type,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "id": type[:3] + "-" + str(uuid.uuid4()),
            "payload": payload
        }

class Status(Update):
    """
    Status message to be sent to the frontend for real-time updates in words.
    """
    def __init__(self, text: str):
        self.text = text
        super().__init__("status")

    def to_frontend(self, conversation_id: str, query_id: str = None):
        return super().to_frontend(
            conversation_id,
            query_id,
            {
                "text": self.text
            }
        )
    
class Warning(Update):
    """
    Warning message to be sent to the frontend.
    """
    def __init__(self, text: str):
        self.text = text
        super().__init__("warning")

    def to_frontend(self, conversation_id: str, query_id: str = None):
        return super().to_frontend(conversation_id, query_id, {
                "text": self.text
        })
    
class Error(Update):
    """
    Error message to be sent to the frontend.
    """
    def __init__(self, text: str):
        self.text = text
        super().__init__("error")

    def to_frontend(self, conversation_id: str, query_id: str = None):
        return super().to_frontend(conversation_id, query_id, {
                "text": self.text
        })

class Completed(Update):
    """
    Completed message to be sent to the frontend (tree is complete all recursions).
    """
    def __init__(self):
        super().__init__("completed")

    def to_frontend(self, conversation_id: str, query_id: str = None):
        return super().to_frontend(conversation_id, query_id, {})

class TreeUpdate(Update):
    """
    Frontend update to represent what nodes have been updated.
    """

    def __init__(self, from_node: str, to_node: str, reasoning: str, last: bool = False):
        """
        from_node: The node that is being updated from.
        to_node: The node that is being updated to.
        reasoning: The reasoning for the update.
        tree_index: The index of the tree being updated.
        last: Whether this is the last update in the branch (whether the tree is complete after this - hardcoded)
        """
        self.from_node = from_node
        self.to_node = to_node
        self.reasoning = reasoning
        self.last = last
        super().__init__("tree_update")

    def to_frontend(self, tree_index: int, conversation_id: str, query_id: str = None, reset: bool = False):
        return super().to_frontend(
            conversation_id,
            query_id,
            {
                "node": self.from_node,
                "tree_index": tree_index,
                "decision": self.to_node,
                "reasoning": self.reasoning,
                "reset": reset
            }
        )