import pytest
import random
from fastapi.responses import JSONResponse

import asyncio
import json

from elysia.api.api_types import (
    ViewPaginatedCollectionData,
    UpdateCollectionMetadataData,
    MetadataNamedVectorData,
    MetadataFieldData,
    InitialiseTreeData,
)
from elysia.api.dependencies.common import get_user_manager
from elysia.api.routes.collections import (
    collections_list,
    view_paginated_collection,
    collection_metadata,
    get_object,
    mapping_types,
    update_metadata,
    delete_metadata,
)
from elysia.api.routes.init import initialise_user, initialise_tree
from elysia.api.core.log import logger, set_log_level
from elysia.preprocessing.collection import preprocess_async
from elysia.util.client import ClientManager

set_log_level("CRITICAL")

from weaviate.classes.config import Configure
import weaviate.classes as wvc
from weaviate.util import generate_uuid5


async def initialise_user_and_tree(user_id: str, conversation_id: str):
    user_manager = get_user_manager()

    response = await initialise_user(
        user_id,
        user_manager,
    )

    response = await initialise_tree(
        user_id,
        conversation_id,
        InitialiseTreeData(
            low_memory=False,
        ),
        user_manager,
    )


def create_named_vectors_collection(
    client_manager: ClientManager, collection_name: str
):

    with client_manager.connect_to_client() as client:

        if client.collections.exists(collection_name):
            client.collections.delete(collection_name)

        collection = client.collections.create(
            name=collection_name,
            properties=[
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


def read_response(response: JSONResponse):
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_collections():

    user_id = "test_user_collections"
    conversation_id = "test_conversation_collections"
    collection_name = "Test_ELYSIA_collection_collections"

    client_manager = ClientManager()
    create_regular_vectorizer_collection(client_manager, collection_name)

    user_manager = get_user_manager()
    await initialise_user_and_tree(user_id, conversation_id)
    basic = await collections_list(user_id, user_manager)
    basic = read_response(basic)
    assert basic["error"] == "", basic["error"]
    assert collection_name in [
        collection["name"] for collection in basic["collections"]
    ]


@pytest.mark.asyncio
async def test_get_object():

    user_id = "test_user_get_object"
    conversation_id = "test_conversation_get_object"
    collection_name = "Test_ELYSIA_collection_get_object"

    client_manager = ClientManager()
    create_regular_vectorizer_collection(client_manager, collection_name)

    user_manager = get_user_manager()
    await initialise_user_and_tree(user_id, conversation_id)

    # first manually get a UUID
    user_local = await user_manager.get_user_local(user_id)
    client_manager = user_local["client_manager"]
    async with client_manager.connect_to_async_client() as client:
        collection = client.collections.get(collection_name)
        response = await collection.query.fetch_objects(limit=1)
        uuid = str(response.objects[0].uuid)

    # test with valid UUID
    basic = await get_object(
        user_id,
        collection_name=collection_name,
        uuid=uuid,
        user_manager=user_manager,
    )

    basic = read_response(basic)
    assert basic["error"] == ""

    # test with invalid UUID
    invalid = await get_object(
        user_id,
        collection_name=collection_name,
        uuid="invalid",
        user_manager=user_manager,
    )

    invalid = read_response(invalid)
    assert invalid["error"] != ""


@pytest.mark.asyncio
async def test_view_paginated_collection():

    user_id = "test_user_view_paginated_collection"
    conversation_id = "test_conversation_view_paginated_collection"
    collection_name = "Test_ELYSIA_collection_view_paginated_collection"
    client_manager = ClientManager()
    create_regular_vectorizer_collection(client_manager, collection_name)

    user_manager = get_user_manager()
    await initialise_user_and_tree(user_id, conversation_id)

    basic = await view_paginated_collection(
        user_id,
        collection_name=collection_name,
        data=ViewPaginatedCollectionData(
            query="",
            page_size=50,
            page_number=1,
            sort_on="title",
            ascending=True,
            filter_config={},
        ),
        user_manager=user_manager,
    )
    basic = read_response(basic)
    assert basic["error"] == ""

    # test with filter
    filter = await view_paginated_collection(
        user_id,
        collection_name=collection_name,
        data=ViewPaginatedCollectionData(
            page_size=50,
            page_number=1,
            sort_on="title",
            ascending=True,
            filter_config={
                "type": "all",
                "filters": [
                    {
                        "field": "product_id",
                        "operator": "equal",
                        "value": "prod2",
                    }
                ],
            },
        ),
        user_manager=user_manager,
    )
    filter = read_response(filter)
    assert filter["error"] == ""

    # test with query
    query = await view_paginated_collection(
        user_id,
        collection_name=collection_name,
        data=ViewPaginatedCollectionData(
            query="hello",
            page_size=50,
            page_number=1,
            ascending=True,
            filter_config={},
        ),
        user_manager=user_manager,
    )
    query = read_response(query)
    assert query["error"] == ""

    # test with out of bounds page number, should return empty list
    out_of_bounds = await view_paginated_collection(
        user_id,
        collection_name=collection_name,
        data=ViewPaginatedCollectionData(
            page_size=50,
            page_number=100,
            sort_on="title",
            ascending=True,
            filter_config={},
        ),
        user_manager=user_manager,
    )
    out_of_bounds = read_response(out_of_bounds)
    assert out_of_bounds["error"] == ""
    assert len(out_of_bounds["items"]) == 0


@pytest.mark.asyncio
async def test_mapping_types():
    await mapping_types()
