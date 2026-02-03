from elysia.tools.retrieval.chunk import Chunker
from elysia.util.parsing import get_estimated_tokens, estimate_tokens


def test_chunker_sentences():
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

    doc = "Hello, world! This is a test. This is another test. " * 20
    chunks, spans = chunker.chunk_by_sentences(
        doc, num_sentences=3, overlap_sentences=1
    )
    assert chunks[0] == "Hello, world! This is a test. This is another test."
    assert chunks[1] == "This is another test. Hello, world! This is a test."


def test_chunker_tokens():
    chunker = Chunker(chunking_strategy="fixed", num_tokens=10)
    doc = "Hello, world! This is a test. This is another test. " * 10
    chunks, spans = chunker.chunk_by_tokens(doc, num_tokens=5, overlap_tokens=0)

    for i, chunk in enumerate(chunks[:-1]):
        assert doc[spans[i][0] : spans[i][1]] == chunk
        assert len(get_estimated_tokens(text=chunk)[0]) == 5

    doc = "Hello, world! This is a test. This is another test. " * 50
    chunks, spans = chunker.chunk_by_tokens(doc, num_tokens=50, overlap_tokens=10)

    for i, chunk in enumerate(chunks[:-1]):
        assert doc[spans[i][0] : spans[i][1]] == chunk
        assert len(get_estimated_tokens(text=chunk)[0]) == 50
