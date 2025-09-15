from elysia import Tool
from elysia.objects import Response

# Import a custom tool from a separate file
from elysia.tools.visualisation.linear_regression import BasicLinearRegression

# Import existing tools
from elysia.tools.retrieval.query import Query
from elysia.tools.retrieval.aggregate import Aggregate
from elysia.tools.text.text import CitedSummarizer, FakeTextResponse


# Or you can define the tool inline here
class TellAJoke(Tool):
    """
    Example tool for testing/demonstration purposes.
    Simply returns a joke as a text response that was an input to the tool.
    """

    def __init__(self, **kwargs):

        # Init requires initialisation of the super class (Tool)
        super().__init__(
            name="tell_a_joke",
            description="Displays a joke to the user.",
            inputs={
                "joke": {
                    "type": str,
                    "description": "A joke to tell.",
                    "required": True,
                }
            },
            end=True,
        )

    # Call must be a async generator function that yields objects to the decision tree
    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):

        # This example tool only returns the input to the tool, so is not very useful
        yield Response(inputs["joke"])

        # You can include more complex logic here via a custom function
