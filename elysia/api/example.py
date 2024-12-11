import os
import sys
import asyncio
sys.path.append(os.getcwd())
os.chdir("../..")

from elysia.api.api_types import GetCollectionData, QueryData, GetCollectionsData, InitialiseTreeData, DebugData
from rich import print

from elysia.api.routes.utils import debug
from elysia.api.routes.query import process
from elysia.api.routes.tree import initialise_tree
from elysia.api.services.tree import TreeManager

import json

class fake_websocket:
    async def send_json(self, data: dict):
        print(data) 
        if data["type"] == "tree_update":
            print(f"connection from {data['payload']['node']} to {data['payload']['decision']}")

tree_manager = TreeManager(
    collection_names=[
        "example_verba_github_issues", 
        "example_verba_slack_conversations", 
        "example_verba_email_chains", 
    ]
)

async def main():

    initialise_tree_data = InitialiseTreeData(
        user_id="test",
        conversation_id="test"
    )

    await initialise_tree(initialise_tree_data, tree_manager)

    query_data = QueryData(
        user_id="test",
        conversation_id="test",
        query="what was kaladin's most recent message?",
        query_id="test"
    )

    await process(query_data.model_dump(), fake_websocket(), tree_manager)

    debug_data = DebugData(
        user_id="test",
        conversation_id="test"
    )

    await debug(debug_data, tree_manager)

if __name__ == "__main__":
    asyncio.run(main())
