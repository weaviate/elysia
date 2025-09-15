import dspy
from elysia.tools.visualisation.objects import (
    BarChart,
    HistogramChart,
    ScatterOrLineChart,
)


class CreateBarChart(dspy.Signature):
    """
    Create one or more bar charts.

    Create a maximum of 9 bar charts.
    Each bar chart should have a maximum of 10 categories.
    Hence each chart should also have a maximum of 10 values per category.
    Pick the most relevant categories and values.
    """

    charts: list[BarChart] = dspy.OutputField(description="The bar chart to create.")
    overall_title: str = dspy.OutputField(
        description=(
            "If providing more than one chart, they will be displayed in a grid. "
            "This is the overall title for above the grid. "
            "Otherwise provide an empty string. "
        ),
    )


class CreateHistogramChart(dspy.Signature):
    """
    Create one or more histogram charts.

    Create a maximum of 9 histogram charts.
    Do not produce more than 50 values per histogram chart. Pick the most relevant values.
    """

    charts: list[HistogramChart] = dspy.OutputField(
        description="The histogram chart to create."
    )
    overall_title: str = dspy.OutputField(
        description=(
            "If providing more than one chart, they will be displayed in a grid. "
            "This is the overall title for above the grid. "
            "Otherwise provide an empty string. "
        ),
    )


class CreateScatterOrLineChart(dspy.Signature):
    """
    Create one or more scatter or line charts.

    Create a maximum of 9 scatter or line charts.
    Create a maximum of 50 points per scatter or line chart. Pick the most relevant points.

    A scatter or line chart can have multiple y-axis values, each with a different label.
    You can combine a line chart with a scatter chart by creating multiple
    """

    charts: list[ScatterOrLineChart] = dspy.OutputField(
        description="The scatter or line chart to create."
    )
    overall_title: str = dspy.OutputField(
        description=(
            "If providing more than one chart, they will be displayed in a grid. "
            "This is the overall title for above the grid. "
            "Otherwise provide an empty string. "
        ),
    )


# class CreateLineChart(dspy.Signature):
#     """
#     Create one or more line charts.

#     Create a maximum of 9 line charts.
#     Create a maximum of 50 points per line chart. Pick the most relevant points.
#     """

#     charts: list[ScatterOrLineChart] = dspy.OutputField(
#         description="The line chart to create."
#     )
#     overall_title: str = dspy.OutputField(
#         description=(
#             "If providing more than one chart, they will be displayed in a grid. "
#             "This is the overall title for above the grid. "
#             "Otherwise provide an empty string. "
#         ),
#     )
