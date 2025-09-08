# self-healing errors can be tested by making a custom tool that returns an error only on thef irst
# and then returns a result on the second call
from elysia import Tree, Tool, Error
from elysia.tree.objects import TreeData
from elysia.util.client import ClientManager
import dspy


class TestSelfHealing(Tool):

    def __init__(self, **kwargs):
        super().__init__(
            name="test_self_healing",
            description="Test self-healing",
            status="healing...",
            inputs={},
            end=False,
            **kwargs,
        )
        self.first_call = True

    async def __call__(
        self,
        tree_data: TreeData,
        inputs: dict,
        base_lm: dspy.LM,
        complex_lm: dspy.LM,
        client_manager: ClientManager,
        **kwargs,
    ):
        if self.first_call:
            yield Error("The first call of this tool _always_ fails.")
            self.first_call = False


def test_self_healing():
    tree = Tree()
    tree.add_tool(TestSelfHealing)

    tree("Call the self-healing tool")
    assert "error" in tree.tree_data.tasks_completed[0]["task"][0]
    assert tree.tree_data.tasks_completed[0]["task"][0]["error"]
    assert "error" not in tree.tree_data.tasks_completed[0]["task"][1]
