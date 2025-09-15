from elysia import Error, Tool, Result
import numpy as np
import matplotlib.pyplot as plt
from elysia.tools.visualisation.objects import (
    ScatterOrLineChart,
    ScatterOrLineDataPoints,
    ScatterOrLineDataPoint,
    ScatterOrLineYAxisData,
    ScatterOrLineChart,
    ChartResult,
)


class BasicLinearRegression(Tool):

    def __init__(self, logger, **kwargs):
        super().__init__(
            name="basic_linear_regression_tool",
            description="""
            Use this tool to perform linear regression between two numeric variables in the environment.
            This also returns a visualisation of the data.
            """,
            status="Running linear regression...",
            inputs={
                "environment_key": {
                    "description": (
                        "A single key (string) of the `environment` dictionary that will be used in the analysis. "
                        "Choose the most relevant key for the analysis according to the user prompt. "
                        "All objects under that key will be used to create the dataframe. "
                    ),
                    "required": True,
                    "type": str,
                    "default": None,
                },
                "x_variable_field": {
                    "description": (
                        "The independent variable for the regression. "
                        "Choose one field title within the `objects` underneath the specific `environment_key`. "
                    ),
                    "required": True,
                    "type": str,
                    "default": None,
                },
                "y_variable_field": {
                    "description": (
                        "The dependent variable for the regression. "
                        "Choose one single field within the `objects` underneath the specific `environment_key`. "
                    ),
                    "required": True,
                    "type": str,
                    "default": None,
                },
            },
            end=False,
        )

    async def __call__(
        self,
        tree_data,
        inputs,
        base_lm,
        complex_lm,
        client_manager,
        **kwargs,
    ):
        environment = tree_data.environment.environment
        environment_key = inputs["environment_key"]
        x_variable_field = inputs["x_variable_field"]
        y_variable_field = inputs["y_variable_field"]

        try:

            # initialise empty matrices to store all objects
            X = np.empty((0, 2))
            y = np.empty((0, 1))

            # iterate over all items in the environment
            for inner_key in environment[environment_key]:

                # convert all objects under this key to a matrix
                inner_X = np.array(
                    [
                        [obj[x_variable_field]]
                        for environment_list in environment[environment_key][inner_key]
                        for obj in environment_list["objects"]
                    ]
                )
                inner_X = np.hstack([np.ones((inner_X.shape[0], 1)), inner_X])
                X = np.vstack([X, inner_X])

                # convert all objects under this key to a matrix
                inner_y = np.array(
                    [
                        [obj[y_variable_field]]
                        for environment_list in environment[environment_key][inner_key]
                        for obj in environment_list["objects"]
                    ]
                )
                y = np.vstack([y, inner_y])

            # calculate the beta hat values via least squares
            beta_hat = np.linalg.inv(X.T @ X + 1e-10 * np.eye(X.shape[1])) @ X.T @ y
            beta_hat_dict = {
                "intercept": beta_hat[0],
                "slope": beta_hat[1],
            }
            pred_y = X @ beta_hat

            # plot the data
            fig, ax = plt.subplots()
            ax.scatter(X[:, 1], y)
            ax.plot(X[:, 1], pred_y, color="red")
            ax.set_title(
                f"Linear regression between {x_variable_field} and {y_variable_field}"
            )
            ax.set_xlabel(x_variable_field)
            ax.set_ylabel(y_variable_field)
            fig.show()

            # yield the result to the decision tree
            yield Result(
                objects=[
                    beta_hat_dict,
                ],
                metadata={
                    "x_variable_field": x_variable_field,
                    "y_variable_field": y_variable_field,
                },
                llm_message=(
                    "Completed linear regression analysis where: "
                    "    x variable: {x_variable_field} "
                    "    y variable: {y_variable_field} "
                    f"    intercept: {beta_hat_dict['intercept']} "
                    f"    slope: {beta_hat_dict['slope']} "
                    "And returned a visualisation of the data."
                ),
            )

            # create the chart payload
            x_axis = [ScatterOrLineDataPoint(value=x) for x in X[:, 1]]
            y_axis = [
                ScatterOrLineYAxisData(
                    label=y_variable_field,
                    kind="scatter",
                    data_points=[ScatterOrLineDataPoint(value=y) for y in y],
                ),
                ScatterOrLineYAxisData(
                    label="Line of best fit",
                    kind="line",
                    data_points=[ScatterOrLineDataPoint(value=y) for y in pred_y],
                ),
            ]
            chart = ScatterOrLineChart(
                title=f"Linear regression between {x_variable_field} and {y_variable_field}",
                description=f"Linear regression between {x_variable_field} and {y_variable_field}",
                x_axis_label=x_variable_field,
                y_axis_label=y_variable_field,
                data=ScatterOrLineDataPoints(x_axis=x_axis, y_axis=y_axis),
            )
            yield ChartResult([chart], "scatter_or_line")

        except Exception as e:
            yield Error(str(e))

    async def is_tool_available(self, tree_data, base_lm, complex_lm, client_manager):
        """
        Available when the 'query' task has been completed and it has added data to the environment.
        """
        return (
            "query" in tree_data.environment.environment
            and len(tree_data.environment.environment["query"]) > 0
        )
