import uuid


def error_payload(text: str, conversation_id: str, query_id: str):
    return {
        "type": "error",
        "id": f"err-{str(uuid.uuid4())}",
        "conversation_id": conversation_id,
        "query_id": query_id,
        "payload": {"text": text},
    }
