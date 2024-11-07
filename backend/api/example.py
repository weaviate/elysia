# import os
# import sys
# sys.path.append(os.getcwd())
# os.chdir("../..")

from backend.api.types import GetCollectionData, ProcessData
from backend.api.app import *

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
