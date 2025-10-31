import pytest
import inspect

from elysia.tools.retrieval.chunk import AsyncCollectionChunker, Chunker
from elysia.util.client import ClientManager

from weaviate.classes.query import QueryReference
from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty
from weaviate.collections.classes.data import DataObject, DataReference
from weaviate.collections import CollectionAsync
from weaviate.collections.classes.internal import Object, QueryReturn
from weaviate.collections.classes.config_vectorizers import _Vectorizer
from weaviate.exceptions import WeaviateInvalidInputError
from weaviate.util import generate_uuid5
from weaviate.client import WeaviateAsyncClient


@pytest.mark.asyncio
async def test_correct_vectoriser_old():
    collection_name_full = "Test_ELYSIA_collection_chunking_vectoriser_full"
    collection_name_named = "Test_ELYSIA_collection_chunking_vectoriser_named"
    client_manager = ClientManager()

    # example data
    data = [
        {
            "random_content_field": (
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
                "Quisque faucibus ex sapien vitae pellentesque sem placerat. "
                "In id cursus mi pretium tellus duis convallis. "
                "Tempus leo eu aenean sed diam urna tempor. "
                "Pulvinar vivamus fringilla lacus nec metus bibendum egestas. "
                "Iaculis massa nisl malesuada lacinia integer nunc posuere. "
                "Ut hendrerit semper vel class aptent taciti sociosqu. "
                "Ad litora torquent per conubia nostra inceptos himenaeos."
            ),
            "other_field": "other_value",
        },
        {
            "random_content_field": (
                "Lorem ipsum dolor sit sit amet consectetur adipiscing elit. "
                "Quisque faucibus ex sapien sed vitae pellentesque sem placerat. "
                "In id cursus mi pretium tellus sed duis convallis. "
                "Tempus leo eu aenean sed diam urna tempor. "
                "Pulvinar vivamus fringilla la cus nec metus bibendum egestas. "
                "Iaculis massa nisl malesuada   lacinia integer nunc posuere. "
                "Ut hendrerit semper vel class ooga aptent taciti sociosqu. "
                "Ad litora torquent per conubia booga nostra inceptos himenaeos."
            ),
            "other_field": "other_value_2",
        },
    ]

    # create collection with vectoriser
    full_vectoriser = Configure.Vectorizer.text2vec_openai(
        model="text-embedding-3-small",
        dimensions=512,
    )
    async with client_manager.connect_to_async_client() as client:
        collection_full = await client.collections.create(
            collection_name_full,
            vectorizer_config=full_vectoriser,
        )
        await collection_full.data.insert_many(data)

    # create collection with named vectoriser
    named_vectoriser = Configure.NamedVectors.text2vec_openai(
        name="content_vector",
        source_properties=["random_content_field"],
        model="text-embedding-3-small",
        dimensions=512,
    )
    async with client_manager.connect_to_async_client() as client:
        collection_named = await client.collections.create(
            collection_name_named,
            vectorizer_config=[
                named_vectoriser,
            ],
        )
        await collection_named.data.insert_many(data)

    try:
        # do chunking on full collection
        collection_chunker = AsyncCollectionChunker(collection_name_full)
        await collection_chunker.create_chunked_reference(
            "random_content_field", client_manager
        )

        async with client_manager.connect_to_async_client() as client:
            # check existence of collection
            assert await client.collections.exists(
                f"ELYSIA_CHUNKED_{collection_name_full.lower()}__"
            )

            # check vectoriser
            collection_full_config = await client.collections.get(
                f"ELYSIA_CHUNKED_{collection_name_full.lower()}__"
            ).config.get()
            assert "default" in collection_full_config.vector_config
            assert (
                collection_full_config.vector_config["default"].vectorizer.vectorizer
                == full_vectoriser.vectorizer
            )

        # check named vectoriser
        collection_chunker = AsyncCollectionChunker(collection_name_named)
        await collection_chunker.create_chunked_reference(
            "random_content_field", client_manager
        )

        async with client_manager.connect_to_async_client() as client:
            # check existence of collection
            assert await client.collections.exists(
                f"ELYSIA_CHUNKED_{collection_name_named.lower()}__"
            )

            # check vectoriser
            collection_named_config = await client.collections.get(
                f"ELYSIA_CHUNKED_{collection_name_named.lower()}__"
            ).config.get()
            assert (
                list(collection_named_config.vector_config.keys())[0]
                in collection_named_config.vector_config
            )
            assert (
                collection_named_config.vector_config[
                    list(collection_named_config.vector_config.keys())[0]
                ].vectorizer.vectorizer
                == named_vectoriser.vectorizer.vectorizer
            )

    finally:
        async with client_manager.connect_to_async_client() as client:
            await client.collections.delete(collection_name_full)
            await client.collections.delete(collection_name_named)
            await client.collections.delete(
                f"ELYSIA_CHUNKED_{collection_name_full.lower()}__"
            )
            await client.collections.delete(
                f"ELYSIA_CHUNKED_{collection_name_named.lower()}__"
            )

        await client_manager.close_clients()


@pytest.mark.asyncio
async def test_correct_vectoriser_new():
    collection_name_full = "Test_ELYSIA_collection_chunking_vectoriser_full_new"
    collection_name_named = "Test_ELYSIA_collection_chunking_vectoriser_named_new"
    client_manager = ClientManager()

    # example data
    data = [
        {
            "random_content_field": (
                "Lorem ipsum dolor sit amet consectetur adipiscing elit. "
                "Quisque faucibus ex sapien vitae pellentesque sem placerat. "
                "In id cursus mi pretium tellus duis convallis. "
                "Tempus leo eu aenean sed diam urna tempor. "
                "Pulvinar vivamus fringilla lacus nec metus bibendum egestas. "
                "Iaculis massa nisl malesuada lacinia integer nunc posuere. "
                "Ut hendrerit semper vel class aptent taciti sociosqu. "
                "Ad litora torquent per conubia nostra inceptos himenaeos."
            ),
            "other_field": "other_value",
        },
        {
            "random_content_field": (
                "Lorem ipsum dolor sit sit amet consectetur adipiscing elit. "
                "Quisque faucibus ex sapien sed vitae pellentesque sem placerat. "
                "In id cursus mi pretium tellus sed duis convallis. "
                "Tempus leo eu aenean sed diam urna tempor. "
                "Pulvinar vivamus fringilla la cus nec metus bibendum egestas. "
                "Iaculis massa nisl malesuada   lacinia integer nunc posuere. "
                "Ut hendrerit semper vel class ooga aptent taciti sociosqu. "
                "Ad litora torquent per conubia booga nostra inceptos himenaeos."
            ),
            "other_field": "other_value_2",
        },
    ]

    # create collection with vectoriser
    full_vectoriser = Configure.Vectors.text2vec_openai(model="text-embedding-3-small")
    async with client_manager.connect_to_async_client() as client:
        collection_full = await client.collections.create(
            collection_name_full,
            vector_config=full_vectoriser,
        )
        await collection_full.data.insert_many(data)

    named_vectoriser = Configure.Vectors.text2vec_openai(
        name="content_vector",
        source_properties=["random_content_field"],
        model="text-embedding-3-small",
    )
    # create collection with named vectoriser
    async with client_manager.connect_to_async_client() as client:
        collection_named = await client.collections.create(
            collection_name_named,
            vector_config=[
                named_vectoriser,
            ],
        )
        await collection_named.data.insert_many(data)

    try:
        # do chunking on full collection
        collection_chunker = AsyncCollectionChunker(collection_name_full)
        await collection_chunker.create_chunked_reference(
            "random_content_field", client_manager
        )

        async with client_manager.connect_to_async_client() as client:
            # check existence of collection
            assert await client.collections.exists(
                f"ELYSIA_CHUNKED_{collection_name_full.lower()}__"
            )

            # check vectoriser
            collection_full_config = await client.collections.get(
                f"ELYSIA_CHUNKED_{collection_name_full.lower()}__"
            ).config.get()
            assert "default" in collection_full_config.vector_config
            assert (
                collection_full_config.vector_config["default"].vectorizer.vectorizer
                == full_vectoriser.vectorizer.vectorizer
            )

        # check named vectoriser
        collection_chunker = AsyncCollectionChunker(collection_name_named)
        await collection_chunker.create_chunked_reference(
            "random_content_field", client_manager
        )

        async with client_manager.connect_to_async_client() as client:
            # check existence of collection
            assert await client.collections.exists(
                f"ELYSIA_CHUNKED_{collection_name_named.lower()}__"
            )

            # check vectoriser
            collection_named_config = await client.collections.get(
                f"ELYSIA_CHUNKED_{collection_name_named.lower()}__"
            ).config.get()
            assert collection_named_config.vector_config[
                list(collection_named_config.vector_config.keys())[0]
            ].vectorizer.source_properties == ["random_content_field"]
            assert (
                collection_named_config.vector_config[
                    list(collection_named_config.vector_config.keys())[0]
                ].vectorizer.vectorizer
                == named_vectoriser.vectorizer.vectorizer
            )

    finally:
        async with client_manager.connect_to_async_client() as client:
            await client.collections.delete(collection_name_full)
            await client.collections.delete(collection_name_named)
            await client.collections.delete(
                f"ELYSIA_CHUNKED_{collection_name_full.lower()}__"
            )
            await client.collections.delete(
                f"ELYSIA_CHUNKED_{collection_name_named.lower()}__"
            )

        await client_manager.close_clients()
