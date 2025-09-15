import asyncio
import os
import pytest
from typing import Any
from weaviate.classes.query import Filter, QueryReference

from elysia.tree.objects import Environment
from elysia.objects import (
    Completed,
    Response,
    Status,
    Text,
    Update,
    Warning,
    Result,
)
from elysia.tools.retrieval.chunk import AsyncCollectionChunker
from elysia.tools.retrieval.objects import (
    ConversationRetrieval,
    DocumentRetrieval,
    MessageRetrieval,
)
from elysia.util.client import ClientManager
from elysia.util.parsing import format_dict_to_serialisable


async def start_client_manager():
    client_manager = ClientManager(
        wcd_url=os.getenv("WCD_URL"),
        wcd_api_key=os.getenv("WCD_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    await client_manager.start_clients()
    return client_manager


async def _retrieve_unchunked_documents(
    collection_name: str, client_manager: ClientManager
):
    async with client_manager.connect_to_async_client() as client:
        collection = client.collections.get(collection_name)
        documents = await collection.query.fetch_objects(
            limit=2,
            return_references=QueryReference(link_on="isChunked"),
        )
    return documents


async def _retrieve_chunked_documents(
    collection_name: str, client_manager: ClientManager
):
    async with client_manager.connect_to_async_client() as client:
        collection = client.collections.get(collection_name)
        documents = await collection.query.fetch_objects(
            limit=2,
            return_references=QueryReference(link_on="fullDocument"),
        )
    return documents


async def _create_chunked_reference(
    collection_name: str, client_manager: ClientManager
):
    collection_chunker = AsyncCollectionChunker(collection_name)
    await collection_chunker.create_chunked_reference("content", client_manager)


async def _chunk_documents(
    collection_name: str, response, client_manager: ClientManager
):
    collection_chunker = AsyncCollectionChunker(collection_name)
    await collection_chunker(response, "content", client_manager)


async def get_collection_information(
    collection_name: str, client_manager: ClientManager
):
    async with client_manager.connect_to_async_client() as client:
        metadata_name = "ELYSIA_METADATA__"
        metadata_collection = client.collections.get(metadata_name)
        metadata = await metadata_collection.query.fetch_objects(
            filters=Filter.by_property("name").equal(collection_name),
            limit=1,
        )
        properties = metadata.objects[0].properties
        format_dict_to_serialisable(properties)
        return properties


async def get_document_retrievals():
    collection_name = "Ml_wikipedia"
    client_manager = await start_client_manager()

    # get the collection information
    collection_information = await get_collection_information(
        collection_name, client_manager
    )
    await _create_chunked_reference(collection_name, client_manager)

    # retrieve some documents
    unchunked_documents = await _retrieve_unchunked_documents(
        collection_name, client_manager
    )

    # chunk the documents
    await _chunk_documents(collection_name, unchunked_documents, client_manager)

    # retrieve the chunked documents
    chunked_documents = await _retrieve_chunked_documents(
        f"ELYSIA_CHUNKED_{collection_name.lower()}__", client_manager
    )

    unchunked_document_properties = [
        doc.properties for doc in unchunked_documents.objects
    ]
    chunked_document_properties = [doc.properties for doc in chunked_documents.objects]

    for i, prop in enumerate(unchunked_document_properties):
        prop["uuid"] = str(unchunked_documents.objects[i].uuid)

    for i, prop in enumerate(chunked_document_properties):
        prop["uuid"] = str(chunked_documents.objects[i].uuid)

    # create a retrieval object
    non_chunked_retrieval = DocumentRetrieval(
        objects=unchunked_document_properties,
        metadata={"collection_name": "ml_wikipedia", "chunked": False},
        mapping=collection_information["mappings"]["document"],
    )
    chunked_retrieval = DocumentRetrieval(
        objects=chunked_document_properties,
        metadata={"collection_name": "ml_wikipedia", "chunked": True},
        mapping=collection_information["mappings"]["document"],
    )
    await chunked_retrieval.async_init(client_manager)
    await client_manager.close_clients()

    return non_chunked_retrieval, chunked_retrieval


@pytest.mark.asyncio
async def test_document_retrieval():
    client_manager = await start_client_manager()

    # if the collection is not found, skip the test
    if not client_manager.client.collections.exists("Ml_wikipedia"):
        pytest.skip("Collection not found in Weaviate")

    non_chunked_retrieval, chunked_retrieval = await get_document_retrievals()

    chunked_json = chunked_retrieval.to_json(mapping=False)

    chunked_fe = await chunked_retrieval.to_frontend(
        "user_id", "conversation_id", "query_id"
    )

    # Check that the document attached to the chunk is >= length of the chunk
    for i, full_doc_obj in enumerate(chunked_fe["payload"]["objects"]):
        assert len(full_doc_obj["content"]) >= len(chunked_json[i]["content"])

    # Check that all chunk spans from the chunk exist in the full documnet
    all_chunk_spans = [
        (obj["uuid"], obj["chunk_spans"])
        for obj in chunked_json
        if obj["chunk_spans"] != []
    ]

    # Find all unique instances of chunkID+chunkSpan
    which_chunk_spans_exist = {
        str(all_chunk_spans[i][0]) + str(all_chunk_spans[i][1]): False
        for i in range(len(all_chunk_spans))
    }

    # Look through full documents (FE payload-objects) and check if the chunk content is present
    for i, full_doc_obj in enumerate(chunked_fe["payload"]["objects"]):
        for chunk_span in full_doc_obj["chunk_spans"]:
            ind = str(chunk_span["uuid"]) + str(
                [chunk_span["start"], chunk_span["end"]]
            )
            if ind in which_chunk_spans_exist:
                which_chunk_spans_exist[ind] = True

    assert all(which_chunk_spans_exist.values())
