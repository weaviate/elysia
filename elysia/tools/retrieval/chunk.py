import asyncio
import inspect
import spacy

from weaviate.classes.config import Configure, DataType, Property, ReferenceProperty
from weaviate.collections.classes.data import DataObject, DataReference
from weaviate.collections import CollectionAsync
from weaviate.collections.classes.internal import Object, QueryReturn
from weaviate.collections.classes.config_vectors import _VectorConfigCreate
from weaviate.exceptions import WeaviateInvalidInputError
from weaviate.util import generate_uuid5
from weaviate.client import WeaviateAsyncClient

from elysia.util.client import ClientManager
from elysia.util.collection import (
    async_get_collection_weaviate_data_types,
)


def chunked_collection_exists(
    collection_name: str, client_manager: ClientManager | None = None
):
    if client_manager is None:
        client_manager = ClientManager()

    with client_manager.connect_to_client() as client:
        return client.collections.exists(f"ELYSIA_CHUNKED_{collection_name.lower()}__")


def delete_chunked_collection(
    collection_name: str, client_manager: ClientManager | None = None
):
    if client_manager is None:
        client_manager = ClientManager()

    with client_manager.connect_to_client() as client:
        client.collections.delete(f"ELYSIA_CHUNKED_{collection_name.lower()}__")


class Chunker:
    def __init__(
        self,
        chunking_strategy: str = "fixed",
        num_tokens: int = 256,
        num_sentences: int = 5,
    ) -> None:
        self.chunking_strategy = chunking_strategy
        assert chunking_strategy in ["fixed", "sentences"]

        self.nlp = spacy.load("en_core_web_sm")

        self.num_tokens = num_tokens
        self.num_sentences = num_sentences

    def count_tokens(self, document: str) -> int:
        return len(self.nlp(document))

    def chunk_by_sentences(
        self,
        document: str,
        num_sentences: int | None = None,
        overlap_sentences: int = 1,
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """
        Given a document (string), return the sentences as chunks and span annotations (start and end indices of chunks).
        Using spaCy to do sentence chunking.
        """
        if num_sentences is None:
            num_sentences = self.num_sentences

        if overlap_sentences >= num_sentences:
            print(
                f"Warning: overlap_sentences ({overlap_sentences}) is greater than num_sentences ({num_sentences}). Setting overlap to {num_sentences - 1}"
            )
            overlap_sentences = num_sentences - 1

        doc = self.nlp(document)
        sentences = list(doc.sents)  # Get sentence boundaries from spaCy

        span_annotations = []
        chunks = []

        i = 0
        while i < len(sentences):
            # Get chunk of num_sentences sentences
            chunk_sentences = sentences[i : i + num_sentences]
            if not chunk_sentences:
                break

            # Get start and end char positions
            start_char = chunk_sentences[0].start_char
            end_char = chunk_sentences[-1].end_char

            # Add chunk and its span annotation
            chunks.append(document[start_char:end_char])
            span_annotations.append((start_char, end_char))

            # Move forward but account for overlap
            i += num_sentences - overlap_sentences

        return chunks, span_annotations

    def chunk_by_tokens(
        self, document: str, num_tokens: int | None = None, overlap_tokens: int = 32
    ) -> tuple[list[str], list[tuple[int, int]]]:
        """
        Given a document (string), return the tokens as chunks and span annotations (start and end indices of chunks).
        Includes overlapping tokens between chunks for better context preservation.
        Uses spaCy for tokenization.
        """
        if num_tokens is None:
            num_tokens = self.num_tokens

        doc = self.nlp(document)
        tokens = list(doc)  # Get tokens from spaCy doc

        span_annotations = []
        chunks = []
        i = 0

        while i < len(tokens):
            # Find end index for current chunk
            end_idx = min(i + num_tokens, len(tokens))
            chunk_tokens = tokens[i:end_idx]

            # Get character spans for the chunk
            start_char = chunk_tokens[0].idx
            end_char = chunk_tokens[-1].idx + len(chunk_tokens[-1])

            # Add chunk and its span annotation
            chunks.append(document[start_char:end_char])
            span_annotations.append((start_char, end_char))

            # Move forward but account for overlap
            i += max(1, num_tokens - overlap_tokens)

        return chunks, span_annotations

    def chunk(self, document: str) -> tuple[list[str], list[tuple[int, int]]]:
        if self.chunking_strategy == "sentences":
            return self.chunk_by_sentences(document)
        elif self.chunking_strategy == "tokens":
            return self.chunk_by_tokens(document)
        else:
            raise ValueError(f"Invalid chunking strategy: {self.chunking_strategy}")


class AsyncCollectionChunker:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.chunker = Chunker("sentences", num_sentences=5)

    async def create_chunked_reference(
        self, content_field: str, client_manager: ClientManager
    ) -> None:
        async with client_manager.connect_to_async_client() as client:
            chunked_collection = await self.get_chunked_collection(
                content_field, client
            )
            full_collection = client.collections.get(self.collection_name)

            # add reference to chunked collection
            try:
                await chunked_collection.config.add_reference(
                    ReferenceProperty(
                        name="fullDocument", target_collection=full_collection.name
                    )
                )
            except WeaviateInvalidInputError:
                # already exists
                pass

            # add reference to full collection
            try:
                await full_collection.config.add_reference(
                    ReferenceProperty(
                        name="isChunked", target_collection=chunked_collection.name
                    )
                )
            except WeaviateInvalidInputError:
                # already exists
                pass

    def get_chunked_collection_name(self) -> str:
        return f"ELYSIA_CHUNKED_{self.collection_name.lower()}__"

    async def chunked_collection_exists(self, client: WeaviateAsyncClient) -> bool:
        return await client.collections.exists(self.get_chunked_collection_name())

    async def get_vectoriser(
        self, content_field: str, client: WeaviateAsyncClient
    ) -> _VectorConfigCreate:
        collection_config = await client.collections.get(
            self.collection_name
        ).config.get()

        # see if any named vectors exclusively vectorise the content field
        if collection_config.vector_config is not None:
            for (
                named_vector_name,
                named_vector_config,
            ) in collection_config.vector_config.items():

                named_vectorizer = named_vector_config.vectorizer

                if (
                    named_vectorizer.source_properties is not None
                    and named_vectorizer.source_properties == [content_field]
                ):
                    try:
                        vectorizer = getattr(
                            Configure.Vectors,
                            named_vectorizer.vectorizer.replace("-", "_"),
                        )

                        valid_args = inspect.signature(vectorizer).parameters
                        return vectorizer(
                            name=named_vector_name,
                            source_properties=[content_field],
                            **{
                                arg: named_vectorizer.model[arg]
                                for arg in named_vectorizer.model
                                if arg in valid_args and arg != "vector_index_config"
                            },
                            vector_index_config=Configure.VectorIndex.hnsw(
                                quantizer=Configure.VectorIndex.Quantizer.sq()  # scalar quantization
                            ),
                        )
                    except AttributeError as e:
                        pass

        # if we haven't returned yet, try the overall vectoriser
        if collection_config.vectorizer_config is not None:
            try:
                vectorizer_name = (
                    collection_config.vectorizer_config.vectorizer.replace("-", "_")
                )
                vectorizer = getattr(Configure.Vectors, vectorizer_name)
                valid_args = inspect.signature(vectorizer).parameters

                return vectorizer(
                    **{
                        arg: collection_config.vectorizer_config.model[arg]
                        for arg in collection_config.vectorizer_config.model
                        if arg in valid_args and arg != "vector_index_config"
                    },
                    vector_index_config=Configure.VectorIndex.hnsw(
                        quantizer=Configure.VectorIndex.Quantizer.sq()  # scalar quantization
                    ),
                )
            except AttributeError as e:
                pass

        # otherwise use default weaviate embedding service
        return Configure.Vectors.text2vec_weaviate(
            dimensions=256,
            vector_index_config=Configure.VectorIndex.hnsw(
                quantizer=Configure.VectorIndex.Quantizer.sq()  # scalar quantization
            ),
        )

    async def get_chunked_collection(
        self, content_field: str, client: WeaviateAsyncClient
    ) -> CollectionAsync:
        if await client.collections.exists(self.get_chunked_collection_name()):
            return client.collections.get(self.get_chunked_collection_name())
        else:
            return await client.collections.create(
                self.get_chunked_collection_name(),
                properties=[
                    Property(name=content_field, data_type=DataType.TEXT),
                    Property(
                        name="chunk_spans",
                        data_type=DataType.INT_ARRAY,
                    ),
                ],
                references=[
                    ReferenceProperty(
                        name="fullDocument", target_collection=self.collection_name
                    )
                ],
                vector_config=await self.get_vectoriser(content_field, client),
            )

    def generate_uuids(
        self,
        chunks: list[str],
        spans: list[tuple[int, int]],
        content_field: str,
    ) -> list[str]:
        chunked_uuids = []
        for i, (chunk, span) in enumerate(zip(chunks, spans)):
            data_object = {content_field: chunk, "chunk_spans": span}
            chunked_uuids.append(generate_uuid5(data_object))
        return chunked_uuids

    async def chunk_single_object(
        self, object: Object, content_field: str
    ) -> tuple[list[str], list[tuple[int, int]], list[str]]:
        content_field_value: str = object.properties[content_field]
        chunks, spans = self.chunker.chunk(content_field_value)
        chunk_uuids = self.generate_uuids(chunks, spans, content_field)
        return chunks, spans, chunk_uuids

    async def chunk_objects_parallel(
        self,
        unchunked_objects: list[Object],
        unchunked_uuids: list[str],
        content_field: str,
    ) -> tuple[dict, dict, dict]:
        tasks = [
            self.chunk_single_object(obj, content_field) for obj in unchunked_objects
        ]

        results = await asyncio.gather(*tasks)

        original_uuid_to_chunks = {}
        original_uuid_to_spans = {}
        original_uuid_to_chunk_uuids = {}

        for uuid, (chunks, spans, chunk_uuids) in zip(unchunked_uuids, results):
            original_uuid_to_chunks[uuid] = chunks
            original_uuid_to_spans[uuid] = spans
            original_uuid_to_chunk_uuids[uuid] = chunk_uuids

        return (
            original_uuid_to_chunks,
            original_uuid_to_spans,
            original_uuid_to_chunk_uuids,
        )

    async def insert_chunks(
        self,
        chunked_collection: CollectionAsync,
        original_uuid_to_chunks: dict,
        original_uuid_to_spans: dict,
        original_uuid_to_chunk_uuids: dict,
        content_field: str,
    ) -> None:
        data_objects = []
        for original_uuid in original_uuid_to_chunks:
            for i, (chunk, span, uuid) in enumerate(
                zip(
                    original_uuid_to_chunks[original_uuid],
                    original_uuid_to_spans[original_uuid],
                    original_uuid_to_chunk_uuids[original_uuid],
                )
            ):
                data_objects.append(
                    DataObject(
                        properties={content_field: chunk, "chunk_spans": span},
                        uuid=uuid,
                        references={"fullDocument": original_uuid},
                    )
                )

        batch_size = 15
        data_objects_batches = [
            data_objects[i : i + batch_size]
            for i in range(0, len(data_objects), batch_size)
        ]

        tasks = [
            chunked_collection.data.insert_many(data_objects)
            for data_objects in data_objects_batches
        ]

        await asyncio.gather(*tasks)

    async def insert_references(
        self,
        full_collection: CollectionAsync,
        original_uuid_to_chunk_uuids: dict,
    ) -> None:
        references = []
        for original_uuid in original_uuid_to_chunk_uuids:
            for chunk_uuid in original_uuid_to_chunk_uuids[original_uuid]:
                references.append(
                    DataReference(
                        from_uuid=original_uuid,
                        from_property="isChunked",
                        to_uuid=chunk_uuid,
                    )
                )

        batch_size = 15
        references_batches = [
            references[i : i + batch_size]
            for i in range(0, len(references), batch_size)
        ]

        tasks = [
            full_collection.data.reference_add_many(references)
            for references in references_batches
        ]

        await asyncio.gather(*tasks)

    def get_chunked_objects(self, objects: list[Object]) -> list[str]:
        """
        Given a list of weaviate objects, find if a reference exists to a chunked object in the separate collection.
        Return the UUIDs of `objects` that have a reference to a chunked object (already have been chunked).
        """
        uuids = []
        for object in objects:
            if (
                object.references is not None
                and "isChunked" in object.references
                and len(object.references["isChunked"].objects) > 0
                and any(
                    [
                        o.collection == self.get_chunked_collection_name()
                        for o in object.references["isChunked"].objects
                    ]
                )
            ):
                uuids.append(str(object.uuid))
        return uuids

    async def __call__(
        self,
        weaviate_response: QueryReturn,
        content_field: str,
        client_manager: ClientManager,
    ) -> None:

        # get all UUIDs of current objects
        all_uuids = [str(object.uuid) for object in weaviate_response.objects]

        # get all UUIDs of chunked objects
        chunked_uuids = self.get_chunked_objects(weaviate_response.objects)

        # get all UUIDs of unchunked objects - these are the objects that need to be chunked
        unchunked_uuids = [
            str(uuid) for uuid in all_uuids if str(uuid) not in chunked_uuids
        ]

        # get all unchunked objects for chunking and inserting
        unchunked_objects = [
            object
            for object in weaviate_response.objects
            if str(object.uuid) in unchunked_uuids
        ]

        # if there are unchunked objects, chunk them
        if len(unchunked_objects) > 0:
            async with client_manager.connect_to_async_client() as client:
                # Get collections
                chunked_collection = await self.get_chunked_collection(
                    content_field, client
                )
                full_collection = client.collections.get(self.collection_name)

                (
                    original_uuid_to_chunks,
                    original_uuid_to_spans,
                    original_uuid_to_chunk_uuids,
                ) = await self.chunk_objects_parallel(
                    unchunked_objects, unchunked_uuids, content_field
                )

                # insert into weaviate
                await self.insert_chunks(
                    chunked_collection,
                    original_uuid_to_chunks,
                    original_uuid_to_spans,
                    original_uuid_to_chunk_uuids,
                    content_field,
                )
                await self.insert_references(
                    full_collection, original_uuid_to_chunk_uuids
                )
