import uuid

from elysia.tree.objects import Objects
from elysia.text.objects import Text

def parse_tree_update(node_id: str, tree_index: int, decision: str, reasoning: str, conversation_id: str, reset: bool):
    return {
        "type": "tree_update",
        "conversation_id": conversation_id,
        "id": "tre-" + str(uuid.uuid4()),
        "payload": {
            "node": node_id,
            "tree_index": tree_index,
            "decision": decision,
            "reasoning": reasoning,
            "reset": reset
        }
    }

def parse_text(text: Text, conversation_id: str):
    return {
        "type": "text",
        "conversation_id": conversation_id,
        "id": "tex-" + str(uuid.uuid4()),
        "payload": text.to_json()
    }

def parse_result(result: Objects, conversation_id: str):
    return {
        "type": "result",
        "conversation_id": conversation_id,
        "id": "res-" + str(uuid.uuid4()),
        "payload": {
            "type": result.type,
            **result.to_json()
        }
    }

def parse_finished(conversation_id: str):
    return {
        "type": "completed",
        "conversation_id": conversation_id,
        "id": "com-" + str(uuid.uuid4()),
        "payload": {}
    }

def parse_error(error: str, conversation_id: str):
    return {
        "type": "error",
        "conversation_id": conversation_id,
        "id": "err-" + str(uuid.uuid4()),
        "payload": {"text": error}
    }

def parse_warning(warning: str, conversation_id: str):
    return {
        "type": "warning",
        "conversation_id": conversation_id,
        "id": "war-" + str(uuid.uuid4()),
        "payload": {"text": warning}
    }