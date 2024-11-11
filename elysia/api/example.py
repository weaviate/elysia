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


query_payload = QueryData(
    query="summarise kaladins last 2 messages",
    user_id="2",
    conversation_id="1"
)

class fake_websocket:
    async def send_json(self, data: dict):
        print(data)

await process(query_payload.dict(), fake_websocket())

tree_manager.get_tree(conversation_id="1", user_id="2").returns