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


class TestMultiBranch:

    @pytest.mark.asyncio
    async def test_incorrect_branch_id(self):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_smart_setup(),
        )

        # error if branch_id is not found
        with pytest.raises(ValueError):
            tree.add_tool(CitedSummarizer, branch_id="incorrect_branch_id")

        # no error if root is True
        tree.add_tool(CitedSummarizer, root=True)

    @pytest.mark.asyncio
    async def test_default_multi_branch(self):

        tree = Tree(
            low_memory=False,
            branch_initialisation="multi_branch",
            settings=Settings.from_smart_setup(),
        )

        assert len(tree.decision_nodes) > 1

        async for result in tree.async_run(
            user_prompt="Hello",
            collection_names=[],
        ):
            pass

    @pytest.mark.asyncio
    async def test_new_multi_branch(self):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_smart_setup(),
        )

        tree.add_branch(
            "respond_to_user",
            "choose between citing objects from the environment, or not citing any objects",
            "choose when responding to user",
            from_branch_id="base",
        )
        tree.add_branch(
            "search_for_objects",
            "choose between searching for objects from the environment, or aggregating objects from the environment",
            "choose when searching for objects",
            from_branch_id="base",
        )

        tree.add_tool(CitedSummarizer, branch_id="respond_to_user")
        tree.add_tool(TextResponse, branch_id="respond_to_user")
        tree.add_tool(Query, branch_id="search_for_objects")
        tree.add_tool(Aggregate, branch_id="search_for_objects")

        assert len(tree.decision_nodes) == 3

        async for result in tree.async_run(
            user_prompt="What was the most recent github issue?",
            collection_names=[],
        ):
            pass


def test_add_tool_with_stem_tool():
    tree = Tree(
        low_memory=False,
        branch_initialisation="empty",
        settings=Settings.from_smart_setup(),
    )

    tree.add_branch(
        branch_id="search",
        instruction="Search for information",
        description="Search for information",
        root=False,
        from_branch_id="base",
    )

    tree.add_tool(TextResponse, branch_id="base")
    tree.add_tool(RetrieveProductData, branch_id="search")
    tree.add_tool(
        CheckResult, branch_id="search", from_tool_ids=["retrieve_product_data"]
    )
    tree.add_tool(
        SendEmail,
        branch_id="search",
        from_tool_ids=["retrieve_product_data", "check_result"],
    )

    response, objects = tree(
        "Retrieve the product data for the product with the id 'prod1', then send an email to danny@weaviate.io with the product data",
        collection_names=[],
    )

    assert "Email sent to danny@weaviate.io!" in response
    assert "Looks good to me!" in response
