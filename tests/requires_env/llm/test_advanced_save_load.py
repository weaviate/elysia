from elysia.tree.tree import Tree
from elysia.tree.util import get_saved_trees_weaviate
from elysia.util.client import ClientManager
from elysia.config import Settings
from elysia.preprocessing import preprocess_async
from elysia.preprocessing.collection import view_preprocessed_collection
import pytest

from deepeval import evaluate, metrics
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from weaviate.util import generate_uuid5
import weaviate.classes as wvc


def create_regular_vectorizer_collection(
    client_manager: ClientManager, collection_name: str
):

    with client_manager.connect_to_client() as client:

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        collection = client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(
                    name="product_id", data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(
                    name="description", data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="tags", data_type=wvc.config.DataType.TEXT_ARRAY
                ),
                wvc.config.Property(name="price", data_type=wvc.config.DataType.NUMBER),
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(
                vectorize_collection_name=False
            ),
        )

        products_data = [
            {
                "product_id": "prod1",
                "title": "Wireless Bluetooth Headphones",
                "description": "High-quality wireless headphones with noise cancellation and 30-hour battery life.",
                "tags": ["electronics", "audio", "wireless", "bluetooth"],
                "price": 199.99,
            },
            {
                "product_id": "prod2",
                "title": "Organic Green Tea",
                "description": "Premium organic green tea sourced from high-altitude tea gardens in Japan.",
                "tags": ["beverage", "tea", "organic", "healthy"],
                "price": 24.99,
            },
            {
                "product_id": "prod3",
                "title": "Yoga Mat with Alignment Lines",
                "description": "Non-slip yoga mat with helpful alignment lines for proper posture during practice.",
                "tags": ["fitness", "yoga", "exercise", "wellness"],
                "price": 39.99,
            },
            {
                "product_id": "prod4",
                "title": "Smart Home Security Camera",
                "description": "WiFi-enabled security camera with motion detection and smartphone notifications.",
                "tags": ["security", "smart-home", "camera", "surveillance"],
                "price": 149.99,
            },
        ]

        with collection.batch.fixed_size(batch_size=10) as batch:
            for product in products_data:
                batch.add_object(
                    properties={
                        "title": product["title"],
                        "description": product["description"],
                        "tags": product["tags"],
                        "price": product["price"],
                    },
                    uuid=generate_uuid5(product["product_id"]),
                )


@pytest.mark.asyncio
async def test_save_load_weaviate():

    client_manager = ClientManager()
    collection_name = "Test_ELYSIA_Product_Data"
    create_regular_vectorizer_collection(client_manager, collection_name)

    # preprocess
    async for result in preprocess_async(collection_name, client_manager, force=True):
        if "error" in result and result["error"] != "":
            assert False, result["error"]

    try:
        settings = Settings.from_smart_setup()
        tree = Tree(settings=settings)

        # ask about edward
        response, objects = tree(
            "What is the description of the Yoga Mat with Alignment Lines",
            collection_names=[
                collection_name,
            ],
        )

        # save the tree to weaviate
        await tree.export_to_weaviate(
            collection_name="Test_ELYSIA_save_load_collection_llm",
            client_manager=client_manager,
        )

        # load the tree from weaviate
        loaded_tree = await Tree.import_from_weaviate(
            collection_name="Test_ELYSIA_save_load_collection_llm",
            conversation_id=tree.conversation_id,
            client_manager=client_manager,
        )

        assert tree.user_id == loaded_tree.user_id
        assert tree.conversation_id == loaded_tree.conversation_id
        assert (
            loaded_tree.tree_data.conversation_history
            == tree.tree_data.conversation_history
        )

        # now ask about his email without referring to the product name
        response, _ = loaded_tree("what about its price?")

        # check that the output includes emails from edward
        # Use a more appropriate metric for the test: check that the response references Edward's email.
        metric = metrics.GEval(
            name="Check the price of the product is referenced",
            criteria=(
                "The response should reference the price of the product. "
                "The answer should be about the price of the product, and should not hallucinate information if not present. "
                "The price of the product should be 39.99"
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
                LLMTestCaseParams.RETRIEVAL_CONTEXT,
            ],
        )

        test_case = LLMTestCase(
            input="what about its price?",
            actual_output=response,
            expected_output="""
            There should be a response about the Yoga Mat with Alignment Lines.
            Ensure that the response talks about the price of the product (39.99 - currency unknown, currency relevant to the test).
            You are only testing if the price of the product is referenced and correct.
            """,
            retrieval_context=[str(obj) for obj in objects],
        )

        res = evaluate(test_cases=[test_case], metrics=[metric])
        for test_case in res.test_results:
            assert test_case.success, test_case.metrics_data[0].reason

    finally:
        async with client_manager.connect_to_async_client() as client:
            if await client.collections.exists(collection_name):
                await client.collections.delete(collection_name)

            if await client.collections.exists("Test_ELYSIA_save_load_collection_llm"):
                await client.collections.delete("Test_ELYSIA_save_load_collection_llm")

        await client_manager.close_clients()
