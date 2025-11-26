import os
from re import S
import pytest
import asyncio
import dspy
from dspy import LM
from elysia.objects import Result
from copy import deepcopy
from elysia import Tool
from elysia.tree.objects import TreeData
from elysia.objects import Response, tool
from elysia.config import Settings, configure
from elysia.tree.tree import Tree
from elysia.tools.text.text import TextResponse
from elysia.util.client import ClientManager
from elysia.tools.retrieval.query import Query
from elysia.tools.retrieval.aggregate import Aggregate

configure(logging_level="CRITICAL")


# == define some tools ==
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


class RunIfTrueFalseTool(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="run_if_true_false_tool",
            description="Always returns False",
        )

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        return False

    async def run_if_true(
        self,
        tree_data: dict,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
    ):
        return False, {}

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
        **kwargs,
    ):

        tree_data.environment.add(
            "rule_tool",
            Result(
                objects=[{"message": "Rule tool called!!!"}],
                payload_type="text",
            ),
        )
        yield False


class RunIfTrueTrueTool(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="run_if_true_true_tool",
            description="Always returns True",
        )

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        return False

    async def run_if_true(
        self,
        tree_data: TreeData,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
    ):
        return True, {}

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
        **kwargs,
    ):

        tree_data.environment.add(
            "rule_tool",
            Result(
                objects=[{"message": "Rule tool called!!!"}],
                payload_type="text",
            ),
        )
        yield True


class RunIfTrueTrueWithInputsTool(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="run_if_true_true_tool",
            description="Always returns True",
            inputs={
                "message": {
                    "type": str,
                    "description": "A test input",
                    "default": "This is the default input",
                }
            },
        )
        self.use_default_inputs = kwargs.get("use_default_inputs", False)

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        return False

    async def run_if_true(
        self,
        tree_data: TreeData,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
    ):
        if self.use_default_inputs:
            return True, {}
        else:
            return True, {"message": "This is not the default input"}

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
        **kwargs,
    ):

        tree_data.environment.add(
            "rule_tool",
            Result(
                objects=[{"message": inputs["message"]}],
                payload_type="text",
            ),
        )
        yield True


class ToolNotAvailable(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="always_pick_this_tool",
            description="No matter what, always pick this tool, and END the conversation.",
            end=True,
        )

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        return False

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
        **kwargs,
    ):
        yield True


class ToolAvailable(Tool):
    def __init__(self, **kwargs):
        super().__init__(
            name="always_pick_this_tool",
            description="No matter what, always pick this tool, and END the conversation.",
            end=True,
        )

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        return True

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: LM,
        complex_lm: LM,
        client_manager: ClientManager,
        **kwargs,
    ):
        yield True


class IncorrectToolInitialisation_kwargs_init(Tool):
    def __init__(self):
        super().__init__(
            name="incorrect_tool_initialisation",
            description="This tool should not be initialised",
        )


class IncorrectToolInitialisation_kwargs_call(Tool):
    def __init__(self):
        super().__init__(
            name="incorrect_tool_initialisation",
            description="This tool should not be initialised",
        )

    async def __call__(self, tree_data, inputs, base_lm, complex_lm, client_manager):
        yield True


class IncorrectToolInitialisation_call_non_async(Tool):
    def __init__(self):
        super().__init__(
            name="incorrect_tool_initialisation",
            description="This tool should not be initialised",
        )

    def __call__(self, tree_data, inputs, base_lm, complex_lm, client_manager):
        return True


class IncorrectToolInitialisation_call_non_async_generator(Tool):
    def __init__(self):
        super().__init__(
            name="incorrect_tool_initialisation",
            description="This tool should not be initialised",
        )

    async def __call__(self, tree_data, inputs, base_lm, complex_lm, client_manager):
        return True


@tool
async def example_async_decorator_tool():
    return "This is a test response from the decorator tool"


async def run_tree(
    user_prompt: str,
    collection_names: list[str],
    tools: list[type[Tool]],
    remove_tools: bool = False,
    **kwargs,
):

    settings = Settings()
    settings.smart_setup()
    settings.configure(
        base_model="gpt-4o-mini",
        base_provider="openai",
        complex_model="gpt-4o",
        complex_provider="openai",
    )

    tree = Tree(
        low_memory=False,
        branch_initialisation="empty",
        settings=settings,
    )

    if not remove_tools:
        tree.add_tool(TextResponse)

    for tool in tools:
        tree.add_tool(tool, **kwargs)

    async for result in tree.async_run(
        user_prompt,
        collection_names=collection_names,
    ):
        pass

    return tree


@pytest.mark.asyncio
async def test_run_if_true_false_tool():

    tree = await run_tree("Hello", [], [RunIfTrueFalseTool])

    assert "rule_tool" not in tree.tree_data.environment.environment


@pytest.mark.asyncio
async def test_run_if_true_true_tool():

    tree = await run_tree("Hello", [], [RunIfTrueTrueTool])

    assert "rule_tool" in tree.tree_data.environment.environment


@pytest.mark.asyncio
async def test_run_if_true_true_with_default_inputs_tool():
    tree = await run_tree(
        "Hello",
        [],
        [RunIfTrueTrueWithInputsTool],
        use_default_inputs=True,
    )

    get = tree.tree_data.environment.get("rule_tool")
    assert get is not None
    assert len(get) == 1
    assert get[0].objects[0]["message"] == "This is the default input"


@pytest.mark.asyncio
async def test_run_if_true_true_with_non_default_inputs_tool():
    tree = await run_tree(
        "Hello",
        [],
        [RunIfTrueTrueWithInputsTool],
        use_default_inputs=False,
    )

    get = tree.tree_data.environment.get("rule_tool")
    assert get is not None
    assert len(get) == 1
    assert get[0].objects[0]["message"] == "This is not the default input"


@pytest.mark.asyncio
async def test_tool_not_available():

    with pytest.raises(ValueError):  # should have no tools available
        await run_tree("Hello", [], [ToolNotAvailable], remove_tools=True)


@pytest.mark.asyncio
async def test_tool_available():

    tree = await run_tree("Hello", [], [ToolAvailable], remove_tools=True)
    all_decision_history = []
    for iteration in tree.decision_history:
        all_decision_history.extend(iteration)

    assert "always_pick_this_tool" in all_decision_history


def test_incorrect_tool_initialisation():
    with pytest.raises(TypeError):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_smart_setup(),
        )
        tree.add_tool(IncorrectToolInitialisation_kwargs_init)

    with pytest.raises(TypeError):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_smart_setup(),
        )
        tree.add_tool(IncorrectToolInitialisation_kwargs_call)

    with pytest.raises(TypeError):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_smart_setup(),
        )
        tree.add_tool(IncorrectToolInitialisation_call_non_async)

    with pytest.raises(TypeError):
        tree = Tree(
            low_memory=False,
            branch_initialisation="empty",
            settings=Settings.from_smart_setup(),
        )
        tree.add_tool(IncorrectToolInitialisation_call_non_async_generator)


@pytest.mark.asyncio
async def test_example_decorator_tool():

    # Async tools should work
    tree = await run_tree(
        "Hello", [], [example_async_decorator_tool], remove_tools=True
    )
    assert "example_async_decorator_tool" in tree.tools

    # Sync tools should not work
    with pytest.raises(TypeError):

        @tool
        def example_sync_decorator_tool():
            return "This is a test response from the decorator tool"


@pytest.mark.asyncio
async def test_example_decorator_tool_from_tree():

    # Async tools should work
    tree = Tree()

    @tool(tree=tree)
    async def example_async_decorator_tool_from_tree():
        return "This is a test response from the decorator tool"

    assert "example_async_decorator_tool_from_tree" in tree.tools


def test_decorator_tool_typed_inputs():

    tree = Tree()

    @tool(tree=tree)
    async def example_decorator_tool(x: int, y: int):
        return x + y

    assert "example_decorator_tool" in tree.tools
    assert "x" in tree.tools["example_decorator_tool"].inputs
    assert "y" in tree.tools["example_decorator_tool"].inputs
    assert tree.tools["example_decorator_tool"].inputs["x"]["type"] is int
    assert tree.tools["example_decorator_tool"].inputs["y"]["type"] is int


def test_decorator_tool_typed_inputs_with_default_inputs():

    tree = Tree()

    @tool(tree=tree)
    async def example_decorator_tool(x: int = 1, y: int = 2):
        return x + y

    assert "example_decorator_tool" in tree.tools

    assert "x" in tree.tools["example_decorator_tool"].inputs
    assert "y" in tree.tools["example_decorator_tool"].inputs
    assert tree.tools["example_decorator_tool"].inputs["x"]["type"] is int
    assert tree.tools["example_decorator_tool"].inputs["y"]["type"] is int
    assert tree.tools["example_decorator_tool"].inputs["x"]["default"] == 1
    assert tree.tools["example_decorator_tool"].inputs["y"]["default"] == 2


def test_decorator_tool_untyped_inputs():

    tree = Tree()

    @tool(tree=tree)
    async def example_decorator_tool(x, y):
        return x + y

    assert "example_decorator_tool" in tree.tools
    assert "x" in tree.tools["example_decorator_tool"].inputs
    assert "y" in tree.tools["example_decorator_tool"].inputs
    assert tree.tools["example_decorator_tool"].inputs["x"]["type"] == "Not specified"
    assert tree.tools["example_decorator_tool"].inputs["y"]["type"] == "Not specified"


def test_decorator_with_elysia_inputs():
    tree = Tree()

    @tool(tree=tree)
    async def example_decorator_tool(
        x: int, y: int, tree_data, base_lm, complex_lm, client_manager
    ):
        return x + y

    assert "example_decorator_tool" in tree.tools
    assert "x" in tree.tools["example_decorator_tool"].inputs
    assert "y" in tree.tools["example_decorator_tool"].inputs
    assert "tree_data" not in tree.tools["example_decorator_tool"].inputs
    assert "base_lm" not in tree.tools["example_decorator_tool"].inputs
    assert "complex_lm" not in tree.tools["example_decorator_tool"].inputs
    assert "client_manager" not in tree.tools["example_decorator_tool"].inputs
