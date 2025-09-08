import asyncio
import pytest
from elysia.util.dummy_adapter import DummyAdapter
from dspy import configure, ChatAdapter, LM
import dspy


@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


# Global variable to track adapter state
_current_adapter_type = None


def get_current_adapter_type():
    """Helper function to check what type of adapter is currently configured"""
    global _current_adapter_type
    return _current_adapter_type


@pytest.fixture(autouse=True, scope="module")
def use_dummy_adapter():
    global _current_adapter_type

    dummy_adapter = DummyAdapter()
    configure(adapter=dummy_adapter)
    _current_adapter_type = "dummy"

    yield

    configure(adapter=ChatAdapter())
    _current_adapter_type = "chat"


if __name__ == "__main__":
    from dspy.adapters import JSONAdapter

    configure(adapter=DummyAdapter())
    print(get_current_adapter_type())

    from elysia.tree.util import DecisionNode
    from elysia.tree.tree import Tree
    from elysia.util.client import ClientManager

    lm = dspy.LM("gpt-4o-mini")

    # class TestSignature(dspy.Signature):
    #     input = dspy.InputField(description="input")
    #     output = dspy.OutputField(description="output")

    # class TestModule(dspy.Module):
    #     def __init__(self):
    #         self.mod = dspy.ChainOfThought(TestSignature)

    #     def forward(self, input: str, lm: dspy.LM):
    #         return self.mod(input=input, lm=lm)

    #     def aforward(self, input: str, lm: dspy.LM):
    #         return self.mod.aforward(input=input, lm=lm)

    # class NonModuleClass:
    #     def __init__(self):
    #         self.mod = TestModule()

    #     def __call__(self, input: str, lm: dspy.LM):
    #         return self.mod(input=input, lm=lm)

    #     async def aforward(self, input: str, lm: dspy.LM):
    #         return await self.mod.aforward(input=input, lm=lm)

    # async def test_non_module_class():
    #     non_module_class = NonModuleClass()
    #     return await non_module_class.aforward(input="hi", lm=lm)

    # print(asyncio.run(test_non_module_class()))

    tree = Tree()

    decision_node = DecisionNode(
        id="test_decision_node",
        instruction="test_instruction",
        options={
            "text_response": {
                "description": "test_description",
                "inputs": {
                    "text": {
                        "description": "response",
                        "type": str,
                        "default": "test_default",
                    }
                },
                "end": True,
                "status": "running",
            }
        },
    )

    async def test_decision_node():

        out = await decision_node(
            tree_data=tree.tree_data,
            base_lm=lm,
            complex_lm=lm,
            available_tools=["text_response"],
            unavailable_tools=[],
            successive_actions={},
            client_manager=ClientManager(),
        )

        return out[0].reasoning

    print(asyncio.run(test_decision_node()))
