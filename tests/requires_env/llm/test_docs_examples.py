from httpx import delete
import pytest
import os


def test_create_tools_simple():

    from elysia import tool

    @tool
    async def add1(x: int, y: int) -> int:
        """
        Return the sum of two numbers.
        """
        return x + y

    from elysia import Tree

    tree = Tree()
    tree.add_tool(add1)

    response, objects = tree.run("What is 1123 + 48332?")

    assert str(1123 + 48332) in response

    tree = Tree()

    @tool(tree=tree, branch_id="base")
    async def add2(x: int, y: int) -> int:
        return x + y

    @tool
    async def calculate_two_numbers(x: int, y: int):
        """
        This function calculates the sum, product, and difference of two numbers.
        """
        yield {
            "sum": x + y,
            "product": x * y,
            "difference": x - y,
        }
        yield f"I just performed some calculations on {x} and {y}."

    tree = Tree()
    tree.add_tool(calculate_two_numbers)

    response, objects = tree.run("What is 1123 + 48332?")

    env = tree.tree_data.environment.environment
    assert "calculate_two_numbers" in env
    assert "default" in env["calculate_two_numbers"]

    result = env["calculate_two_numbers"]["default"][0]
    assert "sum" in result["objects"][0]
    assert result["objects"][0]["sum"] == 1123 + 48332
    assert result["objects"][0]["product"] == 1123 * 48332
    assert result["objects"][0]["difference"] == 1123 - 48332

    from math import prod
    from elysia import Error

    @tool
    async def perform_mathematical_operations(
        numbers: list[int | float], operation: str = "sum"
    ):
        """
        This function calculates a mathematical operation on the `numbers` list.
        The `numbers` input must be a list of integers or floats.
        The `operation` input must be one of: "sum" or "product". These are the only options.
        """

        if operation == "sum":
            yield sum(numbers)
        elif operation == "product":
            yield prod(numbers)
            # This will return an error back to the decision tree
            yield Error(
                f"You picked the input {operation}, but it was not in the available operations: 'sum' or 'product'"
            )
            return  # Then return out of the tool early

        yield f"I just performed a {operation} on {numbers}."

    tree = Tree()
    tree.add_tool(perform_mathematical_operations)

    tree("What is 2379 x 234 x 213 x 3?")

    env = tree.tree_data.environment.environment
    result = env["perform_mathematical_operations"]["default"][0]
    assert result["objects"][0]["tool_result"] == 2379 * 234 * 213 * 3


def test_query_weaviate():

    try:
        from rich import print

        import requests, json

        url = "https://raw.githubusercontent.com/weaviate/weaviate-examples/main/jeopardy_small_dataset/jeopardy_tiny.json"
        resp = requests.get(url)
        data = json.loads(resp.text)

        from elysia.util.client import ClientManager
        from weaviate.classes.config import Configure

        client_manager = ClientManager()

        with client_manager.connect_to_client() as client:

            if client.collections.exists("JeopardyQuestion"):
                client.collections.delete("JeopardyQuestion")

            client.collections.create(
                "JeopardyQuestion",
                vector_config=Configure.Vectors.text2vec_weaviate(),
            )

            jeopardy = client.collections.get("JeopardyQuestion")
            response = jeopardy.data.insert_many(data)

            if response.has_errors:
                print(response.errors)
            else:
                print("Insert complete.")

        from elysia import preprocess

        preprocess("JeopardyQuestion", force=True)

        from elysia import view_preprocessed_collection

        print(view_preprocessed_collection("JeopardyQuestion"))

        from elysia import Tree

        tree = Tree()

        print(tree.view())

        tree.tree

        response, objects = tree(
            "Find a single question about Science",
            collection_names=["JeopardyQuestion"],
        )

        tree("What about animals?")

        print(tree.conversation_history)

    finally:
        client_manager = ClientManager()

        with client_manager.connect_to_client() as client:
            if client.collections.exists("JeopardyQuestion"):
                client.collections.delete("JeopardyQuestion")

            from elysia import delete_preprocessed_collection

            delete_preprocessed_collection("JeopardyQuestion", client_manager)


def test_data_analysis():
    try:
        import sklearn
    except:
        pytest.skip(
            "Skipping data analysis test as sklearn is not installed. "
            "To run this test, install sklearn. "
            "`pip install -U scikit-learn`"
        )

    # Box 1: Configure Elysia
    from elysia import configure

    configure(
        base_model="gemini-2.0-flash-001",  # replace models and providers with which ever LM you want to use
        complex_model="gemini-2.0-flash-001",
        base_provider="openrouter/google",
        complex_provider="openrouter/google",
        wcd_url=os.getenv("WCD_URL"),  # replace with your Weaviate REST endpoint URL
        wcd_api_key=os.getenv(
            "WCD_API_KEY"
        ),  # replace with your Weaviate cloud API key
        openrouter_api_key=os.getenv(
            "OPENROUTER_API_KEY"
        ),  # replace with your OpenAI API key, or whichever API key you will use for your LMs
        logging_level="DEBUG",
    )

    # Box2: Download and Process Data
    from sklearn import datasets

    data = datasets.load_diabetes()
    X, Y = data.data, data.target

    # Box 3: Import to Weaviate
    from elysia.util.client import ClientManager
    import weaviate.classes.config as wvc

    with ClientManager().connect_to_client() as client:
        collection = client.collections.create(
            "ELYSIA_Test_Diabetes", vector_config=wvc.Configure.Vectors.self_provided()
        )

        with collection.batch.dynamic() as batch:
            for i in range(len(X)):
                batch.add_object({"predictor": X[i, 0], "target": Y[i]})

    # Box 4: Preprocess
    from elysia import preprocess

    preprocess("ELYSIA_Test_Diabetes")

    # Box 5: Create Tool
    from elysia import tool
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt

    @tool
    async def fit_linear_regression(env_key, x_var, y_var, collection_name, tree_data):
        """
        Fit a linear regression model to data.
        Requires querying for data first.

        Args:
            env_key: The key of the environment to use (e.g. 'query').
            x_var: Independent variable field name in environment under the key.
            y_var: Dependent variable field name in environment under the key.
        """
        objs = tree_data.environment.find(env_key, collection_name, 0)["objects"]
        X = [[datum.get(x_var)] for datum in objs]
        Y = [datum.get(y_var) for datum in objs]

        model = LinearRegression().fit(X, Y)

        # plt.scatter(X, Y)
        # plt.plot(X, model.predict(X), color="red")
        # plt.show()

        return {
            "intercept": model.intercept_,
            "coef": model.coef_,
            "collection_name": collection_name,
        }

    # Box 6: Run
    from elysia import Tree

    tree = Tree()
    tree.add_tool(fit_linear_regression)
    response, objects = tree(
        "Fit a linear regression on the Diabetes data",
        collection_names=["ELYSIA_Test_Diabetes"],
    )
