# ABOUTME: Service layer for document upload, processing, chunking, and storage in Weaviate.
# ABOUTME: Orchestrates the complete workflow from file upload to searchable document chunks.
import uuid
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional

from elysia.util.document_parser import DocumentParserFactory
from elysia.tools.retrieval.chunk import Chunker, AsyncCollectionChunker
from elysia.util.client import ClientManager
from elysia.preprocessing.collection import preprocess_async
from elysia.api.core.log import logger

from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter


class DocumentService:
    """Service for handling document upload and processing"""

    DOCUMENTS_COLLECTION = "ELYSIA_UPLOADED_DOCUMENTS"

    def __init__(self, client_manager: ClientManager):
        """
        Args:
            client_manager: Weaviate client manager
        """
        self.client_manager = client_manager
        self.chunker = Chunker(chunking_strategy="sentences", num_sentences=5)

    async def upload_document(
        self,
        file_path: Path,
        filename: str,
        user_id: str,
        auto_preprocess: bool = True,
    ) -> dict:
        """
        Complete document upload workflow

        Supported formats: PDF (text-based), TXT, Markdown
        """
        try:
            file_extension = Path(filename).suffix.lower()
            if file_extension not in DocumentParserFactory.supported_extensions():
                raise ValueError(
                    f"Unsupported file type: {file_extension}. "
                    f"Supported: {', '.join(sorted(DocumentParserFactory.supported_extensions()))}"
                )

            parser = DocumentParserFactory.get_parser(file_extension)
            parsed_result = await parser.parse(file_path)

            content = parsed_result["content"]
            element_types = parsed_result["element_types"]
            doc_metadata = parsed_result["metadata"]

            logger.info(
                f"Parsed {filename}: {len(content)} chars, "
                f"element types: {element_types}"
            )

            await self._ensure_collection_exists()

            document_id = str(uuid.uuid4())

            file_size = file_path.stat().st_size
            await self._store_document(
                document_id=document_id,
                content=content,
                filename=filename,
                file_type=file_extension.lstrip('.'),
                file_size=file_size,
                user_id=user_id,
                metadata=doc_metadata,
                element_types=element_types,
            )

            chunks_created = await self._chunk_document(
                document_id=document_id,
                content=content,
            )

            if auto_preprocess:
                try:
                    await self._preprocess_collection()
                except Exception as preprocess_error:
                    logger.warning(
                        f"Preprocessing failed (document still uploaded): {str(preprocess_error)}"
                    )

            logger.info(
                f"Successfully uploaded: {filename} ({chunks_created} chunks)"
            )

            return {
                "success": True,
                "document_id": document_id,
                "collection_name": self.DOCUMENTS_COLLECTION,
                "filename": filename,
                "file_type": file_extension.lstrip('.'),
                "chunks_created": chunks_created,
                "element_types": element_types,
                "message": f"Document '{filename}' uploaded successfully",
            }

        except Exception as e:
            logger.exception(f"Error uploading document: {filename}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to upload: {str(e)}",
            }

    async def _ensure_collection_exists(self) -> None:
        """Create documents collection if it doesn't exist"""
        async with self.client_manager.connect_to_async_client() as client:
            if await client.collections.exists(self.DOCUMENTS_COLLECTION):
                return

            await client.collections.create(
                name=self.DOCUMENTS_COLLECTION,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(
                    model="text-embedding-3-small"
                ),
                properties=[
                    Property(name="document_id", data_type=DataType.TEXT),
                    Property(name="filename", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="file_type", data_type=DataType.TEXT),
                    Property(name="file_size", data_type=DataType.INT),
                    Property(name="user_id", data_type=DataType.TEXT),
                    Property(name="upload_date", data_type=DataType.DATE),
                    Property(name="content_preview", data_type=DataType.TEXT),
                    Property(name="element_types", data_type=DataType.TEXT_ARRAY),
                    Property(
                        name="metadata",
                        data_type=DataType.OBJECT,
                        nested_properties=[
                            Property(name="title", data_type=DataType.TEXT),
                            Property(name="author", data_type=DataType.TEXT),
                            Property(name="filename", data_type=DataType.TEXT),
                            Property(name="filetype", data_type=DataType.TEXT),
                            Property(name="page_count", data_type=DataType.INT),
                        ],
                    ),
                ],
            )
            logger.info(f"Created collection: {self.DOCUMENTS_COLLECTION}")

    async def _store_document(
        self,
        document_id: str,
        content: str,
        filename: str,
        file_type: str,
        file_size: int,
        user_id: str,
        metadata: dict,
        element_types: list,
    ) -> None:
        """Store document in Weaviate collection"""
        async with self.client_manager.connect_to_async_client() as client:
            collection = client.collections.get(self.DOCUMENTS_COLLECTION)

            content_preview = content[:500] + "..." if len(content) > 500 else content

            await collection.data.insert(
                properties={
                    "document_id": document_id,
                    "filename": filename,
                    "content": content,
                    "file_type": file_type,
                    "file_size": file_size,
                    "user_id": user_id,
                    "upload_date": datetime.now(UTC).isoformat().replace('+00:00', 'Z'),
                    "content_preview": content_preview,
                    "element_types": element_types,
                    "metadata": metadata,
                },
                uuid=document_id,
            )

    async def _chunk_document(
        self,
        document_id: str,
        content: str,
    ) -> int:
        """
        Chunk document and store in chunked collection

        Returns:
            Number of chunks created
        """
        collection_chunker = AsyncCollectionChunker(self.DOCUMENTS_COLLECTION)

        await collection_chunker.create_chunked_reference(
            content_field="content",
            client_manager=self.client_manager
        )

        async with self.client_manager.connect_to_async_client() as client:
            chunked_collection = await collection_chunker.get_chunked_collection(
                content_field="content",
                client=client,
            )

            chunks, spans = self.chunker.chunk(content)

            chunk_uuids = collection_chunker.generate_uuids(
                chunks, spans, "content"
            )

            await collection_chunker.insert_chunks(
                chunked_collection=chunked_collection,
                original_uuid_to_chunks={document_id: chunks},
                original_uuid_to_spans={document_id: spans},
                original_uuid_to_chunk_uuids={document_id: chunk_uuids},
                content_field="content",
            )

            full_collection = client.collections.get(self.DOCUMENTS_COLLECTION)
            await collection_chunker.insert_references(
                full_collection=full_collection,
                original_uuid_to_chunk_uuids={document_id: chunk_uuids},
            )

        return len(chunks)

    async def _preprocess_collection(self) -> None:
        """Run preprocessing on documents collection"""
        try:
            async for _ in preprocess_async(
                collection_name=self.DOCUMENTS_COLLECTION,
                client_manager=self.client_manager,
                force=True,
            ):
                pass
            logger.info(f"Preprocessed collection: {self.DOCUMENTS_COLLECTION}")
        except Exception as e:
            logger.error(f"Error preprocessing collection: {e}")

    async def list_user_documents(
        self,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> dict:
        """
        List documents uploaded by user

        Args:
            user_id: User ID
            limit: Max documents to return
            offset: Offset for pagination

        Returns:
            Dictionary with document list
        """
        try:
            async with self.client_manager.connect_to_async_client() as client:
                if not await client.collections.exists(self.DOCUMENTS_COLLECTION):
                    return {
                        "user_id": user_id,
                        "documents": [],
                        "total_count": 0,
                    }

                collection = client.collections.get(self.DOCUMENTS_COLLECTION)

                result = await collection.query.fetch_objects(
                    filters=Filter.by_property("user_id").equal(user_id),
                    limit=limit,
                    offset=offset,
                )

                documents = []
                for obj in result.objects:
                    chunk_count = 0
                    if hasattr(obj, 'references') and obj.references:
                        is_chunked_refs = obj.references.get("isChunked", None)
                        if is_chunked_refs and hasattr(is_chunked_refs, 'objects'):
                            chunk_count = len(is_chunked_refs.objects)

                    upload_date = obj.properties.get("upload_date", "")
                    if isinstance(upload_date, datetime):
                        upload_date = upload_date.isoformat()

                    documents.append({
                        "document_id": obj.properties.get("document_id", ""),
                        "filename": obj.properties.get("filename", ""),
                        "file_type": obj.properties.get("file_type", ""),
                        "file_size": obj.properties.get("file_size", 0),
                        "upload_date": upload_date,
                        "content_preview": obj.properties.get("content_preview", ""),
                        "chunk_count": chunk_count,
                        "element_types": obj.properties.get("element_types", []),
                    })

                return {
                    "user_id": user_id,
                    "documents": documents,
                    "total_count": len(documents),
                }

        except Exception as e:
            logger.exception(f"Error listing documents for user: {user_id}")
            return {
                "user_id": user_id,
                "documents": [],
                "total_count": 0,
                "error": str(e),
            }

    async def delete_document(
        self,
        document_id: str,
        user_id: str,
    ) -> dict:
        """
        Delete document and its chunks

        Args:
            document_id: Document UUID
            user_id: User ID (for authorization)

        Returns:
            Dictionary with deletion result
        """
        try:
            async with self.client_manager.connect_to_async_client() as client:
                if not await client.collections.exists(self.DOCUMENTS_COLLECTION):
                    return {
                        "success": False,
                        "error": "Documents collection does not exist",
                        "message": "No documents have been uploaded yet",
                    }

                collection = client.collections.get(self.DOCUMENTS_COLLECTION)

                doc = await collection.query.fetch_object_by_id(document_id)

                if doc is None:
                    return {
                        "success": False,
                        "error": "Document not found",
                        "message": f"Document with ID '{document_id}' does not exist or has already been deleted",
                    }

                if doc.properties.get("user_id") != user_id:
                    return {
                        "success": False,
                        "error": "Unauthorized",
                        "message": "You are not authorized to delete this document",
                    }

                await collection.data.delete_by_id(document_id)

                logger.info(f"Deleted document: {document_id}")
                return {
                    "success": True,
                    "message": "Document deleted successfully",
                }

        except Exception as e:
            logger.exception(f"Error deleting document: {document_id}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to delete document: {str(e)}",
            }
