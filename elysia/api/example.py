# import os
# import sys
# sys.path.append(os.getcwd())
# os.chdir("../..")

from elysia.api.api_types import GetCollectionData, ProcessData
from elysia.api.app import *

import json

# get collections
collection_names = await collections()
collection_names = json.loads(collection_names.body)["collections"]

# get collection properties
get_collection_payload = GetCollectionData(
    collection_name=collection_names[0]["name"],
    page=1000,
    pageSize=9
)
collection = await get_collection(get_collection_payload)
collection_properties = json.loads(collection.body)["properties"]
items = json.loads(collection.body)["items"]


process_payload = ProcessData(
    user_prompt="what are the most common issues from the verba github issues collection from 2024, sort by the most recent?"
)

class fake_websocket:
    async def send_json(self, data: dict):
        print(data)
        try:
            print(data["payload"]["objects"][0])
        except Exception as e:
            print("didnae work, error: ", e)

await process(process_payload, fake_websocket())