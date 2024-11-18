import os
import sys
import asyncio
sys.path.append(os.getcwd())
os.chdir("../..")

from elysia.api.api_types import GetCollectionData, QueryData, GetCollectionsData
from elysia.api.app import *

import json

# get collections
get_collections_payload = GetCollectionsData(
    user_id="1"
)
collection_names = await collections()
collection_names = json.loads(collection_names.body)["collections"]

# get collection properties
get_collection_payload = GetCollectionData(
    collection_name=collection_names[0]["name"],
    page=1,
    pageSize=9
)
collection = await get_collection(get_collection_payload)
collection_properties = json.loads(collection.body)["properties"]
items = json.loads(collection.body)["items"]


# initialise tree
initialise_tree_payload = InitialiseTreeData(
    user_id="2",
    conversation_id="1"
)
initialise_tree_response = await initialise_tree(initialise_tree_payload)

tree = json.loads(initialise_tree_response.body)["tree"]

query_payload = QueryData(
    query="what did edward say about elephants in slack",
    user_id="2",
    conversation_id="1"
)

class fake_websocket:
    async def send_json(self, data: dict):
        print(data) 
        if data["type"] == "tree_update":
            print(f"connection from {data['payload']['node']} to {data['payload']['decision']}")

await process(query_payload.dict(), fake_websocket())

tree_manager.get_tree(conversation_id="1", user_id="2").returns
tree_manager.get_tree(conversation_id="1", user_id="2").conversation_history

test_return_objects = tree_manager.get_tree(conversation_id="1", user_id="2").returns.retrieved["example_verba_slack_conversations"].objects

for obj in test_return_objects:
    print(obj)

object_relevance_payload = ObjectRelevanceData(
    conversation_id="1",
    user_prompt="what did edward say about elephants in slack",
    objects=test_return_objects
)
object_relevance_response = await object_relevance(object_relevance_payload)
print(json.loads(object_relevance_response.body)["any_relevant"])