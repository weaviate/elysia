from pydantic import BaseModel
from pydantic.fields import Field

from typing import Optional, Literal, Any
from datetime import datetime
from elysia.objects import Result


class BarChartData(BaseModel):
    x_labels: list[str | int | float] = Field(
        description="The shared labels for the x-axis. Must be all string or all numeric. Cannot mix.",
        min_length=1,
        max_length=10,
    )
    y_values: dict[str, list[int | float]] = Field(
        description="A dictionary of data points for the y-axis, with the key being the category/group for the data points.",
        min_length=1,
        max_length=10,
    )


class BarChart(BaseModel):
    title: str
    description: str
    x_axis_label: str
    y_axis_label: str
    data: BarChartData = Field(
        description="A dictionary of bar chart data points, with the key being the category/group for the data points."
    )


class HistogramData(BaseModel):
    distribution: list[float | int]


class HistogramChart(BaseModel):
    title: str
    description: str
    data: dict[str, HistogramData] = Field(
        description="A dictionary of histogram data points, with the key being the category/grouping for the data points."
    )


class ScatterOrLineDataPoint(BaseModel):
    value: int | float | datetime | str
    label: Optional[str] = Field(
        default="",
        description="If a data point is highlighted, you can label it.",
        min_length=1,
        max_length=50,
    )


class ScatterOrLineYAxisData(BaseModel):
    label: str = Field(
        description="The label for the data points going into the y-axis. "
    )
    kind: Literal["scatter", "line"] = Field(
        description="Whether the data points (for this y-axis data) are scatter points, or a line plot. "
    )
    data_points: list[ScatterOrLineDataPoint]


class ScatterOrLineDataPoints(BaseModel):
    x_axis: list[ScatterOrLineDataPoint] = Field(
        default=[],
        description=(
            "The shared x-axis data points. This MUST be populated. "
            "If you think there is no relevant data for the x-axis, "
            "do not leave it empty. X-values are required for Y-values to be plotted. "
        ),
    )
    y_axis: list[ScatterOrLineYAxisData] = Field(
        default=[],
        description=(
            "Each element of the list has a label for the data points going into the y-axis. "
            "Each one is either a set of scatter points, or a line plot. "
            "Which can be combined in a single chart, and will be automatically separated into groups in the same chart (with labels). "
            "There can be multiple sets of data, under different labels, indicating different groups of data. "
            "These will be plotted on the same y-axis, but with different colours, shapes, etc."
        ),
        min_length=1,
        max_length=20,
    )
    normalize_y_axis: Optional[bool] = Field(
        default=False,
        description=(
            "If true, all values in the y-axis will be normalized to the range 0-1. "
            "Useful if the two values are not on the same scale. "
            "Do not modify the labels, or the values, these will be automatically changed."
        ),
    )


class ScatterOrLineChart(BaseModel):
    title: str
    description: str
    x_axis_label: str
    y_axis_label: str
    data: ScatterOrLineDataPoints = Field(
        description="A dictionary of scatter or line data points, with the key being the label for the data points."
    )


class ChartResult(Result):
    def __init__(
        self,
        charts: list[BarChart | HistogramChart | ScatterOrLineChart],
        chart_type: Literal["bar", "histogram", "scatter_or_line"],
        title: str = "",
        metadata: dict[str, Any] = {},
    ):
        if len(charts) == 1 and title == "":
            title = charts[0].title

        super().__init__(
            objects=[chart.model_dump() for chart in charts],
            metadata={"chart_title": title, "chart_type": chart_type, **metadata},
            payload_type=f"{chart_type}_chart",
            mapping=None,
            name=f"{chart_type}_chart",
        )

    def llm_parse(self):
        if "impossible" in self.metadata:
            out = f"Model judged creation of {self.metadata['chart_type']} chart to be impossible for reason: "
            if "impossible_reasoning" in self.metadata:
                out += f"'{self.metadata['impossible_reasoning']}'. "
            out += "Returning to the decision tree..."
            return out
        else:
            return f"Created {len(self.objects)} {self.metadata['chart_type']} chart(s) titled '{self.metadata['chart_title']}'."
