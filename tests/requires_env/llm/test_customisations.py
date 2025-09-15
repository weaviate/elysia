import pytest
from deepeval import evaluate, metrics
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from elysia import Tree, Settings, tool


def test_tree_no_client():

    settings = Settings.from_smart_setup()
    settings.WCD_URL = ""
    settings.WCD_API_KEY = ""

    tree = Tree(
        low_memory=False,
        branch_initialisation="one_branch",
        settings=settings,
    )

    response, objects = tree.run(
        "What was the most recent message in the messages collection?",
        collection_names=["Test_ELYSIA_Messages_collection"],
    )

    metric = metrics.GEval(
        name="Judge of agent being unable to connect to Weaviate",
        criteria="""
        The response should inform the user that the agent is unable to connect to Weaviate,
        and to set the WCD_URL and WCD_API_KEY in the settings.
        OR, the response should inform the user to analyse the data.
        """,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
    )

    test_case = LLMTestCase(
        input="What was the most recent message in the messages collection?",
        actual_output=response,
        expected_output="""
        The agent is unable to connect to Weaviate,
        and to set the WCD_URL and WCD_API_KEY in the settings.
        OR, the response should inform the user to analyse the data.
        """,
    )

    res = evaluate(test_cases=[test_case], metrics=[metric])
    for test_case in res.test_results:
        assert test_case.success, test_case.metrics_data[0].reason


@pytest.mark.asyncio
async def test_tool_decorator():
    @tool
    async def add_two_numbers(x: int, y: int) -> int:
        return x + y

    tree = Tree(
        low_memory=False,
        branch_initialisation="one_branch",
        settings=Settings.from_smart_setup(),
    )
    tree.add_tool(add_two_numbers)

    assert "add_two_numbers" in tree.tools

    prompt = "What is the sum of 1123213 and 2328942390843209?"
    response, objects = tree.run(prompt)

    assert "2328942391966422" in response

    found = False
    for action in tree.actions_called[prompt]:
        if (
            1123213 in action["inputs"].values()
            and 2328942390843209 in action["inputs"].values()
        ):
            found = True
            break

    assert found


@pytest.mark.asyncio
async def test_tool_decorator_with_args():

    tree = Tree(
        low_memory=False,
        branch_initialisation="one_branch",
        settings=Settings.from_smart_setup(),
    )

    @tool(tree=tree)
    async def add_two_numbers(x: int, y: int) -> int:
        return x + y

    assert "add_two_numbers" in tree.tools

    prompt = "What is the sum of 1123213 and 2328942390843209?"
    response, objects = tree.run(prompt)

    assert "2328942391966422" in response

    found = False
    for action in tree.actions_called[prompt]:
        if (
            1123213 in action["inputs"].values()
            and 2328942390843209 in action["inputs"].values()
        ):
            found = True
            break

    assert found

    tree = Tree()

    @tool(tree=tree, status="Adding two numbers", end=True)
    async def add_two_numbers2(x: int, y: int) -> int:
        """
        This function adds two numbers together.
        """
        return x + y
