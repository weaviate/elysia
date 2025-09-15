from elysia.tools.visualisation.objects import (
    BarChart,
    HistogramChart,
    ScatterOrLineChart,
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime
import numpy as np


def convert_chart_types_to_matplotlib(
    charts: list[BarChart | HistogramChart | ScatterOrLineChart],
) -> Figure:
    plt.style.use("seaborn-v0_8-whitegrid")
    n_charts = len(charts)
    n_cols = min(2, n_charts)  # Max 2 columns
    n_rows = (n_charts + n_cols - 1) // n_cols  # Ceiling division

    # Calculate figure size based on number of charts while respecting max size
    fig_width = min(18, 7 * n_cols)
    fig_height = min(15, 5 * n_rows)

    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False
    )

    # colour cycle
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for chart, ax in zip(charts, axs.flatten()):

        if isinstance(chart, BarChart):
            _create_bar_chart(chart, ax, colours)
        elif isinstance(chart, HistogramChart):
            _create_histogram(chart, ax, colours)
        elif isinstance(chart, ScatterOrLineChart):
            _create_scatter_or_line(chart, ax, colours)

        ax.set_title(chart.title)

    # Hide empty subplots
    for i in range(len(charts), len(axs.flatten())):
        axs.flatten()[i].set_visible(False)

    fig.tight_layout()
    return fig


def _create_scatter_or_line(chart: ScatterOrLineChart, ax, colours):

    x_data_points = [dp.value for dp in chart.data.x_axis]
    num_labels = 0
    for i, y_data in enumerate(chart.data.y_axis):

        y_data_points = [d.value for d in y_data.data_points]
        y_data_labels = [d.label for d in y_data.data_points]
        y_data_kind = y_data.kind

        colour = colours[i % len(colours)]

        if (
            chart.data.normalize_y_axis
            and not any(isinstance(dp, str) for dp in y_data_points)
            and not any(isinstance(dp, datetime) for dp in y_data_points)
        ):
            y_data_points = [dp / max(y_data_points) for dp in y_data_points]  # type: ignore

        if y_data_kind == "scatter":
            ax.scatter(x_data_points, y_data_points, label=y_data.label, color=colour)

        elif y_data_kind == "line":
            ax.plot(x_data_points, y_data_points, label=y_data.label, color=colour)

        for j, y_data_label in enumerate(y_data_labels):
            if y_data_label != "":
                ax.scatter(x_data_points[j], y_data_points[j], color=colour)
                if num_labels >= 1 and chart.data.normalize_y_axis:
                    xytext = (0, 10 + 5 * num_labels)
                else:
                    xytext = (0, 10)

                ax.annotate(
                    y_data_label,
                    (x_data_points[j], y_data_points[j]),
                    textcoords="offset pixels",
                    xytext=xytext,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=2),
                )
                num_labels += 1

    ax.set_xlabel(chart.x_axis_label)

    # Rotate x-axis labels to prevent overlap
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if chart.data.normalize_y_axis and (
        "normalized" not in chart.y_axis_label.lower()
        and "normalised" not in chart.y_axis_label.lower()
    ):
        ax.set_ylabel(f"{chart.y_axis_label} (Normalized)")
    else:
        ax.set_ylabel(chart.y_axis_label)

    ax.set_title(chart.title)
    if len(chart.data.y_axis) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def _create_bar_chart(chart: BarChart, ax, colours):
    """Create a bar chart on the given axis."""
    width = 0.8 / len(chart.data.y_values)

    x = np.arange(len(chart.data.x_labels))

    for i, (label, values) in enumerate(chart.data.y_values.items()):
        offset = width * i
        ax.bar(
            x + offset,
            values,
            color=colours[i % len(colours)],
            width=width,
            label=label,
            align="center",
        )

    ax.set_xticks(
        x + width * (len(chart.data.y_values) / 2) * 0.75, chart.data.x_labels
    )
    ax.set_xlabel(chart.x_axis_label)
    ax.set_ylabel(chart.y_axis_label)
    ax.set_title(chart.title)
    if len(chart.data.y_values.keys()) > 1:
        ax.legend(bbox_to_anchor=(1.05, 1))


def _create_histogram(chart: HistogramChart, ax, colours):
    """Create a histogram on the given axis."""

    for i, (label, values) in enumerate(chart.data.items()):
        ax.hist(
            values.distribution, label=label, color=colours[i % len(colours)], alpha=0.7
        )

    if len(chart.data.keys()) == 1:
        ax.set_xlabel(list(chart.data.keys())[0])
    else:
        ax.set_xlabel("Values")

    ax.set_ylabel("Density")
    ax.set_title(chart.title)
    ax.legend()
