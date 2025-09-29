import pytest

from elysia.util.client import ClientManager
from weaviate.classes.config import Configure
from elysia.preprocessing.collection import (
    preprocess_async,
    view_preprocessed_collection_async,
)

# example data
data = [
    {
        "random_content_field": (
            "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
        ),
        "other_field": "other_value",
    },
    {
        "random_content_field": (
            "Lorem ipsum dolor sit sit amet consectetur adipiscing elit. "
        ),
        "other_field": "other_value_2",
    },
]


async def create_no_vectoriser():
    collection_name = "Test_ELYSIA_collection_no_vectoriser"
    client_manager = ClientManager()

    # create collection with no vectoriser
    async with client_manager.connect_to_async_client() as client:
        collection = await client.collections.create(collection_name)
        await collection.data.insert_many(data)


async def create_with_vectoriser_old():
    collection_name = "Test_ELYSIA_collection_vectoriser_old"
    client_manager = ClientManager()

    # create collection with no vectoriser
    async with client_manager.connect_to_async_client() as client:
        collection = await client.collections.create(
            collection_name,
            vectorizer_config=Configure.Vectorizer.text2vec_openai(
                "text-embedding-3-small"
            ),
        )
        await collection.data.insert_many(data)


async def create_with_vectoriser_new():
    collection_name = "Test_ELYSIA_collection_vectoriser_new"
    client_manager = ClientManager()

    # create collection with no vectoriser
    async with client_manager.connect_to_async_client() as client:
        collection = await client.collections.create(
            collection_name,
            vector_config=Configure.Vectors.text2vec_openai(
                model="text-embedding-3-small"
            ),
        )
        await collection.data.insert_many(data)


async def create_with_vectoriser_named_old():
    collection_name = "Test_ELYSIA_collection_vectoriser_named_old"
    client_manager = ClientManager()

    # create collection with no vectoriser
    async with client_manager.connect_to_async_client() as client:
        collection = await client.collections.create(
            collection_name,
            vectorizer_config=[
                Configure.NamedVectors.text2vec_openai(
                    name="test_named_vector", model="text-embedding-3-small"
                )
            ],
        )
        await collection.data.insert_many(data)


async def create_with_vectoriser_named_new():
    collection_name = "Test_ELYSIA_collection_vectoriser_named_new"
    client_manager = ClientManager()

    # create collection with no vectoriser
    async with client_manager.connect_to_async_client() as client:
        collection = await client.collections.create(
            collection_name,
            vector_config=[
                Configure.Vectors.text2vec_openai(
                    name="test_named_vector", model="text-embedding-3-small"
                )
            ],
        )
        await collection.data.insert_many(data)


@pytest.mark.asyncio
async def test_preprocessing_no_vectoriser():
    await create_no_vectoriser()

    async for update in preprocess_async(
        collection_name="Test_ELYSIA_collection_no_vectoriser",
        force=True,
    ):
        if "error" in update and update["error"] != "":
            raise Exception(update["error"])

    properties = await view_preprocessed_collection_async(
        "Test_ELYSIA_collection_no_vectoriser"
    )
    assert properties["vectorizer"] is None
    assert properties["named_vectors"] is None


@pytest.mark.asyncio
async def test_preprocessing_with_vectoriser_old():
    await create_with_vectoriser_old()

    async for update in preprocess_async(
        collection_name="Test_ELYSIA_collection_vectoriser_old",
        force=True,
    ):
        if "error" in update and update["error"] != "":
            raise Exception(update["error"])

    properties = await view_preprocessed_collection_async(
        "Test_ELYSIA_collection_vectoriser_old"
    )
    assert properties["vectorizer"] is not None
    assert properties["vectorizer"]["vectorizer"].lower() == "text2vec_openai"
    assert properties["named_vectors"] is None


@pytest.mark.asyncio
async def test_preprocessing_with_vectoriser_new():
    await create_with_vectoriser_new()

    async for update in preprocess_async(
        collection_name="Test_ELYSIA_collection_vectoriser_new",
        force=True,
    ):
        if "error" in update and update["error"] != "":
            raise Exception(update["error"])

    properties = await view_preprocessed_collection_async(
        "Test_ELYSIA_collection_vectoriser_new"
    )
    assert properties["vectorizer"] is not None
    assert properties["vectorizer"]["vectorizer"].lower() == "text2vec_openai"
    assert properties["named_vectors"] is None


@pytest.mark.asyncio
async def test_preprocessing_with_vectoriser_named_old():
    await create_with_vectoriser_named_old()

    async for update in preprocess_async(
        collection_name="Test_ELYSIA_collection_vectoriser_named_old",
        force=True,
    ):
        if "error" in update and update["error"] != "":
            raise Exception(update["error"])

    properties = await view_preprocessed_collection_async(
        "Test_ELYSIA_collection_vectoriser_named_old"
    )
    assert properties["vectorizer"] is None
    assert properties["named_vectors"] is not None
    assert properties["named_vectors"][0]["vectorizer"].lower() == "text2vec_openai"
    assert properties["named_vectors"][0]["model"] == "text-embedding-3-small"


@pytest.mark.asyncio
async def test_preprocessing_with_vectoriser_named_new():
    await create_with_vectoriser_named_new()

    async for update in preprocess_async(
        collection_name="Test_ELYSIA_collection_vectoriser_named_new",
        force=True,
    ):
        if "error" in update and update["error"] != "":
            raise Exception(update["error"])

    properties = await view_preprocessed_collection_async(
        "Test_ELYSIA_collection_vectoriser_named_new"
    )
    assert properties["vectorizer"] is None
    assert properties["named_vectors"] is not None
    assert properties["named_vectors"][0]["vectorizer"].lower() == "text2vec_openai"
    assert properties["named_vectors"][0]["model"] == "text-embedding-3-small"
