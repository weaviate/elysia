import weaviate
import os

from weaviate.classes.query import Filter
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5  
from rich import print

from backend.util.logging import backend_print

def fetch_items_in_conversation(collection, conversation_id: str):

    items_in_conversation = collection.query.fetch_objects(
        filters=Filter.by_property("conversation_id").equal(conversation_id)
    )
    items_in_conversation = [obj for obj in items_in_conversation.objects]

    return items_in_conversation

def near_text_query_messages(client: weaviate.Client, collection_name: str, query: str, limit: int = 10):

    backend_print(f"Querying collection: {collection_name}")
    collection = client.collections.get(collection_name)
    response = collection.query.near_text(
        query = query,
        limit = limit
    )

    returned_objects = [None] * len(response.objects)
    for i, o in enumerate(response.objects):
        items_in_conversation = fetch_items_in_conversation(collection, o.properties["conversation_id"])
        to_return = [{
            k: v for k, v in item.properties.items()
        } for item in items_in_conversation]
        to_return.sort(key = lambda x: int(x["message_index"]))
        returned_objects[i] = to_return

    return returned_objects


# example usage
if __name__ == "__main__":
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url = os.environ.get("WCD_URL"),
        auth_credentials = Auth.api_key(os.environ.get("WCD_API_KEY")),
        headers = {"X-OpenAI-API-Key": os.environ.get("OPENAI_API_KEY")}
    )
    returned_objects = near_text_query_messages(client, "example_verba_slack_conversations", "What is the weather in Tokyo?", limit = 1)
    print(returned_objects[0])
