from backend.tree.objects import Objects

def parse_decision(decision: str, conversation_id: str, id: str, instruction: str, tree: dict):
    return {
        "type": "decision",
        "conversation_id": conversation_id,
        "payload": {
            "id": id,
            "decision": decision,
            "instruction": instruction,
            "tree": tree
        }
    }


def parse_result(result: Objects, conversation_id: str):
    return {
        "type": "result",
        "conversation_id": conversation_id,
        "payload": {
            "type": result.type,
            **result.to_json()
        }
    }