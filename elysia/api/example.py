import os
import sys
import asyncio
sys.path.append(os.getcwd())
os.chdir("../..")

from elysia.api.api_types import GetCollectionData, QueryData, GetCollectionsData
from elysia.api.app import *
from rich import print

import json

class fake_websocket:
    async def send_json(self, data: dict):
        print(data) 
        if data["type"] == "tree_update":
            print(f"connection from {data['payload']['node']} to {data['payload']['decision']}")

