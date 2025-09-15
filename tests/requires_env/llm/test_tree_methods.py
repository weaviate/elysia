import pytest
import elysia
from elysia.tree.tree import Tree
from elysia.util.client import ClientManager
from elysia.preprocessing import preprocess
from weaviate.util import generate_uuid5
import weaviate.classes as wvc


@pytest.fixture(scope="module")
def create_and_process_collection():
    collection_name = "Test_ELYSIA_Product_Data"
    client_manager = ClientManager()
    create_regular_vectorizer_collection(client_manager, collection_name)
    preprocess(collection_name, client_manager)
    yield
    with client_manager.connect_to_client() as client:
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)
        client_manager.close_clients()


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


def test_various_methods(create_and_process_collection):
    collection_name = "Test_ELYSIA_Product_Data"
    tree = Tree(
        low_memory=False,
    )
    user_prompt = "Find me some bluetooth headphones"
    _, _ = tree(
        user_prompt,
        collection_names=[
            collection_name,
        ],
    )

    # check the user prompt is in the conversation history
    assert tree.tree_data.conversation_history[0]["content"] == user_prompt
    assert tree.tree_data.conversation_history[0]["role"] == "user"

    # check some variables are set
    assert tree.tree_data.num_trees_completed > 0
    assert tree.tree_data.user_prompt == user_prompt

    all_decision_history = []
    for iteration in tree.decision_history:
        all_decision_history.extend(iteration)
    assert len(all_decision_history) > 0
    assert "query" in all_decision_history

    # this should have returned something in the environment
    assert "query" in tree.tree_data.environment.environment
    for env_name in tree.tree_data.environment.environment["query"]:
        for returned_object in tree.tree_data.environment.environment["query"][
            env_name
        ]:
            assert "metadata" in returned_object
            assert len(returned_object["objects"]) > 0

    # this should have updated tasks_completed
    assert len(tree.tree_data.tasks_completed) > 0
    prompt_found = False
    for task in tree.tree_data.tasks_completed:
        if task["prompt"] == user_prompt:
            prompt_found = True
            assert len(task["task"]) > 0
            break

    assert prompt_found

    # check the tree resets
    tree.soft_reset()

    # These should not change
    assert len(tree.tree_data.conversation_history) > 0
    assert len(tree.tree_data.tasks_completed) > 0
    assert (
        "query" in tree.tree_data.environment.environment
    )  # environment should not be reset

    # these should be reset
    assert tree.tree_data.num_trees_completed == 0
    assert tree.tree_index == 1

    all_decision_history = []
    for iteration in tree.decision_history:
        all_decision_history.extend(iteration)
    assert len(all_decision_history) == 0


def test_get_follow_up_suggestions(
    create_and_process_collection,
):  # Added fixture parameter
    tree = Tree()
    _, _ = tree("Find me some bluetooth headphones")
    suggestions = tree.get_follow_up_suggestions(
        context="make it about headphones specifically",
        num_suggestions=5,
    )

    assert len(suggestions) == 5
