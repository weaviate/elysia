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


def test_chunker():
    chunker = Chunker(chunking_strategy="sentences", num_sentences=1)
    doc = "Hello, world! This is a test."
    chunks, spans = chunker.chunk(doc)
    assert len(chunks) == 2
    assert spans == [(0, 13), (14, 29)]
    assert chunks[0] == "Hello, world!"
    assert chunks[1] == "This is a test."
    assert doc[spans[0][0] : spans[0][1]] == chunks[0]
    assert doc[spans[1][0] : spans[1][1]] == chunks[1]

    doc = "Hello, world! This is a test. This is another test."
    chunks, spans = chunker.chunk(doc)
    assert len(chunks) == 3
    assert spans == [(0, 13), (14, 29), (30, 51)]
    assert chunks[0] == "Hello, world!"
    assert chunks[1] == "This is a test."
    assert chunks[2] == "This is another test."
    assert doc[spans[0][0] : spans[0][1]] == chunks[0]
    assert doc[spans[1][0] : spans[1][1]] == chunks[1]
    assert doc[spans[2][0] : spans[2][1]] == chunks[2]
