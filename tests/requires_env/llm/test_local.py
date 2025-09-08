import pytest
import weaviate

from elysia.util.client import ClientManager
from elysia import preprocess

import weaviate
from weaviate.util import generate_uuid5
import weaviate.classes as wvc


def create_and_process_collection(client_manager: ClientManager):
    collection_name = "Test_ELYSIA_Product_Data"
    create_regular_vectorizer_collection(client_manager, collection_name)
    preprocess(collection_name, client_manager)


def create_regular_vectorizer_collection(
    client_manager: ClientManager, collection_name: str
):

    with client_manager.connect_to_client() as client:

        if not client.collections.exists(collection_name):
            collection = client.collections.create(
                name=collection_name,
                properties=[
                    wvc.config.Property(
                        name="product_id", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="title", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="description", data_type=wvc.config.DataType.TEXT
                    ),
                    wvc.config.Property(
                        name="tags", data_type=wvc.config.DataType.TEXT_ARRAY
                    ),
                    wvc.config.Property(
                        name="price", data_type=wvc.config.DataType.NUMBER
                    ),
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
async def test_local_with_tree():

    HOST = "localhost"
    HTTP_PORT = 8080
    GRPC_PORT = 50051

    try:
        client = weaviate.connect_to_local(
            host=HOST,
            port=HTTP_PORT,
            grpc_port=GRPC_PORT,
            skip_init_checks=True,
        )
    except Exception:
        pytest.skip("Local Weaviate not reachable on localhost:8080")

    client_manager = ClientManager(
        wcd_url=HOST,
        wcd_api_key="",
        weaviate_is_local=True,
        local_weaviate_port=HTTP_PORT,
        local_weaviate_grpc_port=GRPC_PORT,
    )

    # try creating a collection and preprocessing with local
    create_and_process_collection(client_manager)

    # try running a tree with local
    from elysia import Tree

    tree = Tree()

    async for _ in tree.async_run(
        "Find me some bluetooth headphones",
        client_manager=client_manager,
        collection_names=["Test_ELYSIA_Product_Data"],
    ):
        pass

    # query tool was successful
    query_found = False
    for action in tree.actions_called["Find me some bluetooth headphones"]:
        if action["name"] == "query":
            query_found = True
            break
    assert query_found

    # objects were found
    # assert len(objects[0]) > 0
