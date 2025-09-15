import pytest

from fastapi.responses import JSONResponse

import json
import os
import random
from elysia.api.api_types import ProcessCollectionData, SaveConfigUserData
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.init import initialise_user
from elysia.api.routes.processor import process_collection
from elysia.api.routes.collections import collection_metadata
from elysia.util.client import ClientManager
from elysia.api.core.log import logger, set_log_level
from elysia.preprocessing.collection import (
    delete_preprocessed_collection_async,
    preprocessed_collection_exists_async,
)
from elysia.api.routes.user_config import get_current_user_config, save_config_user
from elysia.api.routes.collections import update_metadata, delete_metadata
from weaviate.classes.config import Configure
from weaviate.classes.query import Filter
from weaviate.util import generate_uuid5
import weaviate.classes as wvc

from elysia.api.api_types import (
    ViewPaginatedCollectionData,
    UpdateCollectionMetadataData,
    MetadataNamedVectorData,
    MetadataFieldData,
    InitialiseTreeData,
)

set_log_level("CRITICAL")


def create_named_vectors_collection(
    client_manager: ClientManager, collection_name: str
):

    with client_manager.connect_to_client() as client:

        # Delete collection if it exists
        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        # Create collection with multiple named vectors
        collection = client.collections.create(
            name=collection_name,
            properties=[
                wvc.config.Property(
                    name="product_id", data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(name="title", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(name="content", data_type=wvc.config.DataType.TEXT),
                wvc.config.Property(
                    name="category", data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(name="author", data_type=wvc.config.DataType.TEXT),
            ],
            vector_config=[
                wvc.config.Configure.Vectors.text2vec_openai(
                    name="content_vector", source_properties=["content"]
                ),
                wvc.config.Configure.Vectors.text2vec_openai(
                    name="title_vector", source_properties=["title"]
                ),
                wvc.config.Configure.Vectors.self_provided(name="custom_vector"),
            ],
        )

        # Toy data for named vectors collection
        documents_data = [
            {
                "product_id": "doc1",
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "category": "Education",
                "author": "Dr. Smith",
                "custom_vector": [0.1, 0.2, 0.3, 0.4, 0.5],  # Custom embedding
            },
            {
                "product_id": "doc2",
                "title": "Python Programming Basics",
                "content": "Python is a versatile programming language used in web development, data science, and automation.",
                "category": "Programming",
                "author": "Jane Doe",
                "custom_vector": [0.6, 0.7, 0.8, 0.9, 1.0],
            },
            {
                "product_id": "doc3",
                "title": "Database Design Principles",
                "content": "Good database design involves normalization, indexing, and understanding relationships between entities.",
                "category": "Database",
                "author": "Bob Johnson",
                "custom_vector": [0.2, 0.4, 0.6, 0.8, 0.3],
            },
        ]

        # Import data with named vectors
        with collection.batch.fixed_size(batch_size=10) as batch:
            for doc in documents_data:
                batch.add_object(
                    properties={
                        "title": doc["title"],
                        "content": doc["content"],
                        "category": doc["category"],
                        "author": doc["author"],
                    },
                    vector={"custom_vector": doc["custom_vector"]},
                    uuid=generate_uuid5(doc["product_id"]),
                )


def create_regular_vectorizer_collection(
    client_manager: ClientManager, collection_name: str
):

    with client_manager.connect_to_client() as client:

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        collection = client.collections.create(
            name=collection_name,
            properties=[
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


user_manager = get_user_manager()


def read_response(response: JSONResponse):
    return json.loads(response.body)


class fake_websocket:
    def __init__(self):
        self.results = []

    async def send_json(self, data: dict):
        self.results.append(data)


@pytest.mark.asyncio
async def test_process_no_api_keys():
    """
    Test processing a collection with no api keys.
    """

    user_id = "test_user_Process_no_api_keys"
    collection_name = "Test_ELYSIA_collection_process_no_api_keys"

    await initialise_user(
        user_id,
        user_manager,
    )
    local_user = await user_manager.get_user_local(user_id)
    client_manager = ClientManager()
    create_regular_vectorizer_collection(client_manager, collection_name)

    try:
        # get config
        response = await get_current_user_config(user_id, user_manager)
        response = read_response(response)
        config_id = response["config"]["id"]
        config_name = response["config"]["name"]

        # remove api keys
        await save_config_user(
            user_id,
            config_id,
            SaveConfigUserData(
                name=config_name,
                default=True,
                config={
                    "settings": {
                        "BASE_MODEL": "gemini-2.0-flash-001",
                        "BASE_PROVIDER": "openrouter/google",
                        "COMPLEX_MODEL": "gemini-2.0-flash-001",
                        "COMPLEX_PROVIDER": "openrouter/google",
                        "WCD_URL": os.getenv("WCD_URL"),
                        "WCD_API_KEY": os.getenv("WCD_API_KEY"),
                        "API_KEYS": {},
                    }
                },
                frontend_config={},
            ),
            user_manager,
        )

        websocket = fake_websocket()
        await process_collection(
            ProcessCollectionData(
                user_id=user_id, collection_name=collection_name
            ).model_dump(),
            websocket,
            user_manager,
        )

        error_found = False
        for result in websocket.results:
            error_found = result["error"] != ""
        assert error_found

    finally:

        # delete the collections
        try:
            with client_manager.connect_to_client() as client:

                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)

                if await preprocessed_collection_exists_async(
                    collection_name, client_manager
                ):
                    await delete_preprocessed_collection_async(
                        collection_name, client_manager
                    )

        except Exception as e:
            print(f"Can't delete collection: {str(e)}")
            pass


@pytest.mark.asyncio
async def test_process_collection():

    user_id = "test_user_Process_collection"
    collection_name = "Test_ELYSIA_collection_process_collection"

    client_manager = ClientManager()
    create_regular_vectorizer_collection(client_manager, collection_name)
    try:
        await initialise_user(
            user_id,
            user_manager,
        )
        local_user = await user_manager.get_user_local(user_id)
        client_manager = local_user["client_manager"]

        websocket = fake_websocket()
        await process_collection(
            ProcessCollectionData(
                user_id=user_id, collection_name=collection_name
            ).model_dump(),
            websocket,
            user_manager,
        )
        for i, result in enumerate(websocket.results):
            if result["error"] != "":
                assert (
                    False
                ), f"({i}) Message i-1: {websocket.results[i-1]['message']}, error: {result['error']}"

        for result in websocket.results[:-1]:
            assert result["type"] == "update"
            assert result["progress"] < 1

        assert websocket.results[-1]["type"] == "completed"
        assert websocket.results[-1]["progress"] == 1

        with client_manager.connect_to_client() as client:
            assert client.collections.exists(collection_name)
            assert client.collections.exists("ELYSIA_METADATA__")
            metadata_collection = client.collections.get("ELYSIA_METADATA__")
            metadata = metadata_collection.query.fetch_objects(
                filters=Filter.by_property("name").equal(collection_name),
                limit=1,
            )
            assert len(metadata.objects) >= 1

    finally:

        # delete the collections
        try:
            with client_manager.connect_to_client() as client:

                if client.collections.exists(collection_name):
                    client.collections.delete(collection_name)

                if await preprocessed_collection_exists_async(
                    collection_name, client_manager
                ):
                    await delete_preprocessed_collection_async(
                        collection_name, client_manager
                    )

        except Exception as e:
            print(f"Can't delete collection: {str(e)}")
            pass


@pytest.mark.asyncio
async def test_collection_metadata():

    user_id = "test_user_collection_metadata"
    conversation_id = "test_conversation_collection_metadata"
    collection_name = "Test_ELYSIA_collection_collection_metadata"

    client_manager = ClientManager()
    user_manager = get_user_manager()
    create_named_vectors_collection(client_manager, collection_name)

    await initialise_user(
        user_id,
        user_manager,
    )

    try:

        websocket = fake_websocket()
        await process_collection(
            ProcessCollectionData(
                user_id=user_id, collection_name=collection_name
            ).model_dump(),
            websocket,
            user_manager,
        )

        basic = await collection_metadata(
            user_id,
            collection_name=collection_name,
            user_manager=user_manager,
        )
        basic = read_response(basic)
        assert basic["error"] == "", basic["error"]

        assert "summary" in basic["metadata"]
        assert "fields" in basic["metadata"]
        assert "mappings" in basic["metadata"]
        assert "length" in basic["metadata"]
        assert "mappings" in basic["metadata"]
        assert "index_properties" in basic["metadata"]
        assert "named_vectors" in basic["metadata"]
        assert basic["metadata"]["named_vectors"] is not None
        assert len(basic["metadata"]["named_vectors"]) == 3
        assert all(nv["enabled"] for nv in basic["metadata"]["named_vectors"])

        named_vector_names = [nv["name"] for nv in basic["metadata"]["named_vectors"]]
        assert "content_vector" in named_vector_names
        assert "title_vector" in named_vector_names
        assert "custom_vector" in named_vector_names

        assert "vectorizer" in basic["metadata"]
        assert basic["metadata"]["vectorizer"] is None

    finally:
        async with client_manager.connect_to_async_client() as client:
            if await client.collections.exists(collection_name):
                await client.collections.delete(collection_name)


@pytest.mark.asyncio
async def test_update_metadata():

    user_id = "test_user_update_metadata"
    conversation_id = "test_conversation_update_metadata"

    collection_name = "Test_ELYSIA_collection_update_metadata"
    client_manager = ClientManager()
    user_manager = get_user_manager()
    create_named_vectors_collection(client_manager, collection_name)

    try:
        await initialise_user(
            user_id,
            user_manager,
        )

        # preprocess
        websocket = fake_websocket()
        await process_collection(
            ProcessCollectionData(
                user_id=user_id, collection_name=collection_name
            ).model_dump(),
            websocket,
            user_manager,
        )

        # retrieve the metadata
        metadata = await collection_metadata(
            user_id=user_id,
            collection_name=collection_name,
            user_manager=user_manager,
        )
        metadata = read_response(metadata)
        assert metadata["error"] == ""
        metadata = metadata["metadata"]

        # record values
        old_named_vectors = metadata["named_vectors"]
        old_summary = metadata["summary"]
        old_mappings = metadata["mappings"]
        old_fields = metadata["fields"]

        # what to update
        new_named_vectors = [
            MetadataNamedVectorData(
                name="content_vector",
                enabled=True,
                description="Test description for content vector.",
            )
        ]

        new_summary = "This is a test summary."
        new_mappings = {
            "product": {
                "name": "title",
                "category": "content",
                "description": "author",
            }
        }
        new_fields = [
            MetadataFieldData(
                name="title",
                description="A test description for title.",
            )
        ]

        # update the metadata
        update_metadata_response = await update_metadata(
            user_id=user_id,
            collection_name=collection_name,
            data=UpdateCollectionMetadataData(
                named_vectors=new_named_vectors,
                summary=new_summary,
                mappings=new_mappings,
                fields=new_fields,
            ),
            user_manager=user_manager,
        )
        update_metadata_response = read_response(update_metadata_response)
        assert update_metadata_response["error"] == "", update_metadata_response[
            "error"
        ]

        # retrieve the metadata again
        metadata = await collection_metadata(
            user_id=user_id,
            collection_name=collection_name,
            user_manager=user_manager,
        )
        metadata = read_response(metadata)
        assert metadata["error"] == ""
        metadata = metadata["metadata"]

        # check the values have been updated
        for named_vector in metadata["named_vectors"]:
            if named_vector["name"] == "content_vector":
                assert named_vector["enabled"] == True
                assert (
                    named_vector["description"]
                    == "Test description for content vector."
                )
                break

        assert metadata["summary"] == new_summary
        assert "product" in metadata["mappings"]
        for field in metadata["fields"]:
            if field["name"] == "title":
                assert field["description"] == new_fields[0].description
                break

        # check the other named vectors have not changed
        for named_vector in metadata["named_vectors"]:
            if named_vector["name"] != "content_vector":
                assert named_vector["enabled"]
                assert named_vector["description"] == ""
    finally:
        async with client_manager.connect_to_async_client() as client:
            if await client.collections.exists(collection_name):
                await client.collections.delete(collection_name)


@pytest.mark.asyncio
async def test_delete_metadata():
    user_id = "test_user_delete_metadata"
    conversation_id = "test_conversation_delete_metadata"
    collection_name = "Test_ELYSIA_collection_delete_metadata"

    user_manager = get_user_manager()
    client_manager = ClientManager()

    await initialise_user(user_id, user_manager)

    # create a collection
    create_named_vectors_collection(client_manager, collection_name)

    # preprocess
    websocket = fake_websocket()
    await process_collection(
        ProcessCollectionData(
            user_id=user_id, collection_name=collection_name
        ).model_dump(),
        websocket,
        user_manager,
    )
    for i, result in enumerate(websocket.results):
        if result["error"] != "":
            assert (
                False
            ), f"({i}) Message i-1: {websocket.results[i-1]['message']}, error: {result['error']}"

    # check the metadata has been created
    metadata = await collection_metadata(
        user_id=user_id,
        collection_name=collection_name,
        user_manager=user_manager,
    )
    metadata = read_response(metadata)
    assert metadata["error"] == "", metadata["error"]
    assert metadata["metadata"] is not {} and metadata["metadata"] is not None

    # delete the metadata
    delete_metadata_response = await delete_metadata(
        user_id=user_id,
        collection_name=collection_name,
        user_manager=user_manager,
    )
    delete_metadata_response = read_response(delete_metadata_response)
    assert delete_metadata_response["error"] == "", delete_metadata_response["error"]

    # check that the metadata has been deleted
    metadata = await collection_metadata(
        user_id=user_id,
        collection_name=collection_name,
        user_manager=user_manager,
    )
    metadata = read_response(metadata)
    assert metadata["error"] != ""
