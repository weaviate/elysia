import os
import sys
import asyncio
sys.path.append(os.getcwd())
os.chdir("../..")

from elysia.api.api_types import GetCollectionData, QueryData, GetCollectionsData
from elysia.api.app import *
from elysia.tree import complex_lm, base_lm
from rich import print

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
    query="retrieve edward's last message",
    query_id="whatduhek",
    user_id="2",
    conversation_id="1"
)


class fake_websocket:
    async def send_json(self, data: dict):
        print(data) 
        if data["type"] == "tree_update":
            print(f"connection from {data['payload']['node']} to {data['payload']['decision']}")

await process(query_payload.dict(), fake_websocket())



query_payload = QueryData(
    query="what was the conversation attached to that?",
    query_id="whatduhek",
    user_id="2",
    conversation_id="1"
)
await process(query_payload.dict(), fake_websocket())

# test_return_objects = tree_manager.get_tree(conversation_id="1", user_id="2").returns.retrieved["ecommerce"].objects

# for obj in test_return_objects:
#     print(obj)

# object_relevance_payload = ObjectRelevanceData(
#     user_id="2",
#     conversation_id="1",
#     query_id = query_payload.query_id,
#     objects=test_return_objects
# )
# object_relevance_response = await object_relevance(object_relevance_payload)
# print(json.loads(object_relevance_response.body)["any_relevant"])



# query_payload2 = QueryData(
#     query="can you give me more t-shirts around that price point?",
#     query_id="whatduhek2",
#     user_id="2",
#     conversation_id="1"
# )

# class fake_websocket:
#     async def send_json(self, data: dict):
#         print(data) 
#         if data["type"] == "tree_update":
#             print(f"connection from {data['payload']['node']} to {data['payload']['decision']}")

# await process(query_payload2.dict(), fake_websocket())


# test_return_objects = tree_manager.get_tree(conversation_id="1", user_id="2").returns.retrieved["example_verba_github_issues"].objects

# for obj in test_return_objects:
#     print(obj)

# object_relevance_payload = ObjectRelevanceData(
#     user_id="2",
#     conversation_id="1",
#     query_id = query_payload2.query_id,
#     objects=test_return_objects
# )
# object_relevance_response = await object_relevance(object_relevance_payload)
# print(json.loads(object_relevance_response.body)["any_relevant"])