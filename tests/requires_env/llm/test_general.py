import os
import pytest
import asyncio
import dspy
from dspy import LM
from elysia.objects import Result
from copy import deepcopy
from elysia import Tool
from elysia.tree.objects import TreeData
from elysia.config import Settings, configure
from elysia.tree.tree import Tree
from elysia.tools.text.text import TextResponse, CitedSummarizer
from elysia.tools.retrieval.query import Query
from elysia.tools.retrieval.aggregate import Aggregate
from elysia.objects import Response, Result

try:
    # get_ipython
    os.chdir("../..")
except NameError:
    pass

from elysia.util.client import ClientManager


class RetrieveProductData(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="retrieve_product_data",
            description="Retrieve product data from the product data collection.",
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        yield Result(
            objects=[
                {
                    "product_id": "prod1",
                    "title": "Wireless Bluetooth Headphones",
                    "description": "High-quality wireless headphones with noise cancellation and 30-hour battery life.",
                    "tags": ["electronics", "audio", "wireless", "bluetooth"],
                    "price": 199.99,
                },
            ]
        )


class CheckResult(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="check_result",
            description="Check the result of the previous tool.",
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        yield Response("Looks good to me!")


class SendEmail(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="send_email",
            description="Send an email.",
            inputs={
                "email_address": {
                    "type": str,
                    "description": "The email address to send the email to.",
                    "required": True,
                }
            },
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        yield Response(f"Email sent to {inputs['email_address']}!")


@pytest.mark.asyncio
async def test_incorrect_branch_id():
    tree = Tree(
        low_memory=False,
        branch_initialisation="empty",
        settings=Settings.from_smart_setup(),
    )

    # error if branch_id is not found
    with pytest.raises(ValueError):
        tree.add_tool(CitedSummarizer, from_node_id="incorrect_branch_id")

    # no error if from_node_id is not specified
    tree.add_tool(CitedSummarizer)


@pytest.mark.asyncio
async def test_default_multi_branch():

    tree = Tree(
        low_memory=False,
        branch_initialisation="multi_branch",
        settings=Settings.from_smart_setup(),
    )

    assert len([node for node in tree.nodes.values() if node.branch]) > 1

    async for result in tree.async_run(
        user_prompt="Hello",
        collection_names=[],
    ):
        pass


@pytest.mark.asyncio
async def test_new_multi_branch():
    tree = Tree(
        low_memory=False,
        branch_initialisation="empty",
        settings=Settings.from_smart_setup(),
    )
    assert len(tree.nodes) == 1
    base_node_id = list(tree.nodes.values())[0].id
    assert len(tree.nodes[base_node_id].options) == 0

    tree.add_branch(
        "respond_to_user",
        "choose between citing objects from the environment, or not citing any objects",
        "choose when responding to user",
        from_node_id=base_node_id,
        node_id="respond_to_user_id",
    )
    tree.add_branch(
        "search_for_objects",
        "choose between searching for objects from the environment, or aggregating objects from the environment",
        "choose when searching for objects",
        from_node_id=base_node_id,
        node_id="search_for_objects_id",
    )

    cited_summariser_id = tree.add_tool(
        CitedSummarizer, from_node_id="respond_to_user_id"
    )
    text_response_id = tree.add_tool(TextResponse, from_node_id="respond_to_user_id")
    query_id = tree.add_tool(Query, from_node_id="search_for_objects_id")
    aggregate_id = tree.add_tool(Aggregate, from_node_id="search_for_objects_id")

    assert len(tree.nodes) == 7
    assert sorted(tree.nodes["respond_to_user_id"].options) == sorted(
        [cited_summariser_id, text_response_id]
    )
    assert sorted(tree.nodes["search_for_objects_id"].options) == sorted(
        [query_id, aggregate_id]
    )

    async for result in tree.async_run(
        user_prompt="What was the most recent github issue?",
        collection_names=[],
    ):
        pass
