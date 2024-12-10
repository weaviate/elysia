import spacy

from weaviate.classes.config import Property, DataType, ReferenceProperty, Configure

from weaviate.exceptions import WeaviateInvalidInputError
from weaviate.util import generate_uuid5

from elysia.util.collection_metadata import get_collection_weaviate_data_types
from elysia.globals.weaviate_client import client

from tqdm.auto import tqdm

class Chunker:
    def __init__(
        self,
        chunking_strategy: str = "fixed",
        num_tokens: int = 256,
        num_sentences: int = 5
    ):
        self.chunking_strategy = chunking_strategy
        assert chunking_strategy in ["fixed", "sentences"]

        self.nlp = spacy.load("en_core_web_sm")

        self.num_tokens = num_tokens
        self.num_sentences = num_sentences

    def count_tokens(self, document):
        return len(self.nlp(document))

    def chunk_by_sentences(self, document, num_sentences=None, overlap_sentences=1):
        """
        Given a document (string), return the sentences as chunks and span annotations (start and end indices of chunks).
        Using spaCy to do sentence chunking.
        """
        if num_sentences is None:
            num_sentences = self.num_sentences

        if overlap_sentences >= num_sentences:
            print(f"Warning: overlap_sentences ({overlap_sentences}) is greater than num_sentences ({num_sentences}). Setting overlap to {num_sentences - 1}")
            overlap_sentences = num_sentences - 1

        doc = self.nlp(document)
        sentences = list(doc.sents)  # Get sentence boundaries from spaCy
        
        span_annotations = []
        chunks = []
        
        i = 0
        while i < len(sentences):
            # Get chunk of num_sentences sentences
            chunk_sentences = sentences[i:i + num_sentences]
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

    def chunk_by_tokens(self, document, num_tokens=None, overlap_tokens=32):
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

    def chunk(self, document):
        if self.chunking_strategy == "sentences":
            return self.chunk_by_sentences(document)
        elif self.chunking_strategy == "tokens":
            return self.chunk_by_tokens(document)
        

class CollectionChunker:

    def __init__(self, collection_name):
        self.collection_name = collection_name
        self.collection = client.collections.get(collection_name)
        self.chunker = Chunker("sentences", num_sentences=5)

    def create_chunked_reference(self, content_field: str):
        if not self.chunked_collection_exists():
            self.get_chunked_collection(content_field);
        try:
            self.collection.config.add_reference(
                ReferenceProperty(
                    name="isChunked",
                    target_collection=f"ELYSIA_CHUNKED_{self.collection_name}__"
                )
            )
        except WeaviateInvalidInputError:
            pass

    def chunked_collection_exists(self):
        return client.collections.exists(f"ELYSIA_CHUNKED_{self.collection_name}__")
    
    def get_chunked_collection(self, content_field: str):

        # get properties of main collection
        data_types = get_collection_weaviate_data_types(self.collection_name)

        if client.collections.exists(f"ELYSIA_CHUNKED_{self.collection_name}__"):
            return client.collections.get(f"ELYSIA_CHUNKED_{self.collection_name}__")
        else:
            return client.collections.create(
                f"ELYSIA_CHUNKED_{self.collection_name}__",
                properties = [
                    Property(
                        name=content_field,
                        data_type=DataType.TEXT
                    ),
                    Property(
                        name="chunk_spans",
                        data_type=DataType.INT_ARRAY,
                        nested_data_type=DataType.INT
                    )
                ] + [
                    Property(
                        name=name,
                        data_type=data_type
                    )
                    for name, data_type in data_types.items() if name != content_field
                ], # add all properties except the content field TODO: this should be a reference to the original so no need to store twice
                references = [
                    ReferenceProperty(
                        name="fullDocument",
                        target_collection=self.collection_name
                    )
                ],
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-large",
                    dimensions=256
                ),
                vector_index_config=Configure.VectorIndex.hnsw(
                    quantizer=Configure.VectorIndex.Quantizer.sq() # scalar quantization
                ),
            )
    
    def generate_uuids(self, chunks: list[str], spans: list[tuple[int, int]], properties, content_field: str):
        chunked_uuids = []
        for i, (chunk, span) in enumerate(zip(chunks, spans)):
            data_object = {
                content_field: chunk,
                "chunk_spans": span,
                **properties
            }
            chunked_uuids.append(generate_uuid5(data_object))
        return chunked_uuids

    def insert_chunks(self, chunked_collection, chunks, spans, chunked_uuids, properties, content_field: str):
        with chunked_collection.batch.dynamic() as batch:
            for i, (chunk, span) in enumerate(zip(chunks, spans)):
                data_object = {
                    content_field: chunk,
                    "chunk_spans": span,
                    **properties[i]
                }
                batch.add_object(data_object, uuid=chunked_uuids[i])
    
    def insert_references(self, chunked_collection, full_collection, both_uuids):
        """
        Both UUIDS: {original_uuid: [chunked_uuids], ...}
        keys are the original and list is the chunked UUIDs corresponding to the original
        """
        with chunked_collection.batch.dynamic() as chunked_batch:
            for original_uuid in both_uuids:
                for chunked_uuid in both_uuids[original_uuid]:
                    chunked_batch.add_reference(
                        from_uuid = chunked_uuid,
                        from_property="fullDocument",
                        to=original_uuid
                    )
        
        with full_collection.batch.dynamic() as full_batch:
            for original_uuid in both_uuids:
                for chunked_uuid in both_uuids[original_uuid]:
                    full_batch.add_reference(
                        from_uuid = original_uuid,
                        from_property="isChunked",
                        to=chunked_uuid
                    )

    def insert_references_to_original_collection(self, collection, chunked_uuids, original_uuids):
        with collection.batch.dynamic() as batch:
            for chunked_uuid, original_uuid in zip(chunked_uuids, original_uuids):
                batch.add_reference(
                    from_uuid = original_uuid,
                    from_property="isChunked",
                    to=chunked_uuid
                )

    def get_chunked_objects(self, objects):
        """
        Given a list of weaviate objects, find if a reference exists to a chunked object in the separate collection.
        Return the UUIDs of `objects` that have a reference to a chunked object.
        """
        uuids = []
        for object in objects.objects:
            if (
                object.references is not None and 
                "isChunked" in object.references and
                len(object.references["isChunked"].objects) > 0
            ):
                uuids.append(object.uuid)
        return uuids


    def __call__(self, objects, content_field: str):

        # always create the chunked reference, if it already exists, it will be skipped
        # self.create_chunked_reference(content_field)

        # get all UUIDs of current objects
        all_uuids = [object.uuid for object in objects.objects]

        # get all UUIDs of chunked objects
        chunked_uuids = self.get_chunked_objects(objects)

        # get all UUIDs of unchunked objects - these are the objects that need to be chunked
        unchunked_uuids = [uuid for uuid in all_uuids if uuid not in chunked_uuids]

        # get all unchunked objects for chunking and inserting
        unchunked_objects = [object for object in objects.objects if object.uuid in unchunked_uuids]

        assert len(unchunked_objects) == len(unchunked_uuids)

        # if there are unchunked objects, chunk them
        if len(unchunked_objects) > 0:

            # get the chunked collection to insert into
            chunked_collection = self.get_chunked_collection(content_field)

            # keeping track of the chunks/spans, output uuids for each chunk object, and other properties from the collection
            all_chunks = []
            all_spans = []
            chunk_uuids = []
            all_uuids = {}
            all_properties = []
            for i, object in enumerate(unchunked_objects):
                
                # chunk the text and get spans
                chunks, spans = self.chunker.chunk(object.properties[content_field])

                # get other properties from the collection for this object
                other_properties = {
                    name: object.properties[name]
                    for name in object.properties.keys() if name != content_field
                }
                chunk_uuids_i = self.generate_uuids(chunks, spans, other_properties, content_field)

                # a single dict of properties for all chunks corresponding to the same original object
                all_properties.extend([other_properties] * len(chunks)) 

                # all chunks and spans
                all_chunks.extend(chunks)
                all_spans.extend(spans)
                
                # chunk_uuids (list of uuids corresponding to all_chunks), all_uuids (dict of original uuids to chunked uuids, mapping)
                chunk_uuids.extend(chunk_uuids_i)
                all_uuids[unchunked_uuids[i]] = chunk_uuids_i

            # insert into weaviate
            self.insert_chunks(chunked_collection, all_chunks, all_spans, chunk_uuids, all_properties, content_field)
            self.insert_references(chunked_collection, self.collection, all_uuids)
                
