import uuid

from elysia.tree.objects import Objects
from elysia.text.objects import Text

def parse_tree_update(node_id: str, tree_index: int, decision: str, reasoning: str, conversation_id: str, reset: bool, query_id: str = None):
    return {
        "type": "tree_update",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "id": "tre-" + str(uuid.uuid4()),
        "payload": {
            "node": node_id,
            "tree_index": tree_index,
            "decision": decision,
            "reasoning": reasoning,
            "reset": reset
        }
    }

def parse_text(text: Text, conversation_id: str, query_id: str = None):
    return {
        "type": "text",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "id": "tex-" + str(uuid.uuid4()),
        "payload": text.to_json()
    }

def parse_result(result: Objects, code: str, conversation_id: str, query_id: str = None):
    return {
        "type": "result",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "id": "res-" + str(uuid.uuid4()),
        "payload": {
            "type": result.type,
            "code": code,
            **result.to_json()
        }
    }

def parse_finished(conversation_id: str, query_id: str = None):
    return {
        "type": "completed",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "id": "com-" + str(uuid.uuid4()),
        "payload": {}
    }

def parse_error(error: str, conversation_id: str, query_id: str = None):
    return {
        "type": "error",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "id": "err-" + str(uuid.uuid4()),
        "payload": {"text": error}
    }

def parse_warning(warning: str, conversation_id: str, query_id: str = None):
    return {
        "type": "warning",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "id": "war-" + str(uuid.uuid4()),
        "payload": {"text": warning}
    }