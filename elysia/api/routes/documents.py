# ABOUTME: FastAPI routes for document upload, listing, and deletion endpoints.
# ABOUTME: Handles file uploads with validation, temp file management, and user authorization.
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from elysia.api.core.log import logger
from elysia.api.dependencies.common import get_user_manager
from elysia.api.services.user import UserManager
from elysia.api.services.document import DocumentService
from elysia.util.client import ClientManager
from elysia.util.document_parser import DocumentParserFactory


router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024


@router.post("/{user_id}/upload")
async def upload_document(
    user_id: str,
    file: UploadFile = File(...),
    auto_preprocess: bool = Form(True),
    user_manager: UserManager = Depends(get_user_manager),
) -> JSONResponse:
    """
    Upload and process a document (PDF, Word, Markdown, or Text)

    Args:
        user_id: The ID of the user uploading the document
        file: The file to upload
        auto_preprocess: Whether to automatically preprocess the collection (default: True)
        user_manager: The user manager dependency

    Returns:
        JSONResponse with upload results
    """
    logger.info(f"Document upload request from user: {user_id}, file: {file.filename}")

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager: ClientManager = user_local["client_manager"]

        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")

        file.file.seek(0, 2)
        file_size = file.file.tell()
        file.file.seek(0)

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / 1024 / 1024}MB"
            )

        if file_size == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in DocumentParserFactory.supported_extensions():
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported: {', '.join(sorted(DocumentParserFactory.supported_extensions()))}"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = Path(temp_file.name)

        try:
            document_service = DocumentService(client_manager)
            result = await document_service.upload_document(
                file_path=temp_file_path,
                filename=file.filename,
                user_id=user_id,
                auto_preprocess=auto_preprocess,
            )

            if result["success"]:
                return JSONResponse(
                    content=result,
                    status_code=200,
                )
            else:
                return JSONResponse(
                    content=result,
                    status_code=400,
                )

        finally:
            if temp_file_path.exists():
                temp_file_path.unlink()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error in upload_document endpoint")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "message": f"Upload failed: {str(e)}",
            },
            status_code=500,
        )


@router.get("/{user_id}/list")
async def list_documents(
    user_id: str,
    limit: int = 100,
    offset: int = 0,
    user_manager: UserManager = Depends(get_user_manager),
) -> JSONResponse:
    """
    List documents uploaded by user

    Args:
        user_id: The ID of the user
        limit: Maximum number of documents to return (default: 100)
        offset: Offset for pagination (default: 0)
        user_manager: The user manager dependency

    Returns:
        JSONResponse with list of documents
    """
    logger.debug(f"List documents request for user: {user_id}")

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager: ClientManager = user_local["client_manager"]

        document_service = DocumentService(client_manager)
        result = await document_service.list_user_documents(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        logger.exception(f"Error in list_documents endpoint")
        return JSONResponse(
            content={
                "user_id": user_id,
                "documents": [],
                "total_count": 0,
                "error": str(e),
            },
            status_code=500,
        )


@router.delete("/{user_id}/delete/{document_id}")
async def delete_document(
    user_id: str,
    document_id: str,
    user_manager: UserManager = Depends(get_user_manager),
) -> JSONResponse:
    """
    Delete a document and its chunks

    Args:
        user_id: The ID of the user
        document_id: The UUID of the document to delete
        user_manager: The user manager dependency

    Returns:
        JSONResponse with deletion result
    """
    logger.info(f"Delete document request: user={user_id}, document={document_id}")

    try:
        user_local = await user_manager.get_user_local(user_id)
        client_manager: ClientManager = user_local["client_manager"]

        document_service = DocumentService(client_manager)
        result = await document_service.delete_document(
            document_id=document_id,
            user_id=user_id,
        )

        if result["success"]:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=400)

    except Exception as e:
        logger.exception(f"Error in delete_document endpoint")
        return JSONResponse(
            content={
                "success": False,
                "error": str(e),
                "message": f"Deletion failed: {str(e)}",
            },
            status_code=500,
        )


@router.get("/supported-formats")
async def supported_formats() -> JSONResponse:
    """
    Get list of supported file formats

    Returns:
        JSONResponse with supported formats
    """
    return JSONResponse(
        content={
            "supported_extensions": sorted(list(DocumentParserFactory.supported_extensions())),
            "description": "Supported document formats: PDF (text-based, no OCR), TXT, Markdown",
            "parsers": {
                "pdf": "pypdf - extracts text from text-based PDFs",
                "txt": "native - UTF-8 text files",
                "md": "native - Markdown files"
            },
            "max_file_size_mb": MAX_FILE_SIZE / 1024 / 1024,
        },
        status_code=200,
    )
