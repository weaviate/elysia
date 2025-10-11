# ABOUTME: Simple, robust document parsers for PDF, TXT, and Markdown files.
# ABOUTME: Uses pypdf for PDFs and native file reading for text-based formats.
from pathlib import Path
from typing import Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

from elysia.api.core.log import logger


class PDFParser:
    """
    Simple PDF parser using pypdf library

    Reliable text extraction from text-based PDFs.
    Does not handle scanned PDFs (OCR) or complex layouts.
    """

    SUPPORTED_EXTENSIONS = {".pdf"}

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=2)

    @classmethod
    def is_supported(cls, file_extension: str) -> bool:
        """Check if file type is supported"""
        return file_extension.lower() in cls.SUPPORTED_EXTENSIONS

    async def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF file and extract text"""
        result = await asyncio.get_event_loop().run_in_executor(
            self._executor,
            self._parse_sync,
            file_path
        )
        return result

    def _parse_sync(self, file_path: Path) -> Dict[str, Any]:
        """Synchronous PDF parsing"""
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf library not installed. Install with: pip install pypdf"
            )

        try:
            reader = PdfReader(str(file_path))

            content_parts = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    content_parts.append(text.strip())

            full_content = "\n\n".join(content_parts)

            if not full_content or len(full_content.strip()) < 10:
                raise ValueError(
                    "PDF appears to be empty or contains no extractable text. "
                    "It may be a scanned document (OCR not supported)."
                )

            metadata = {
                "filename": file_path.name,
                "filetype": "pdf",
                "page_count": len(reader.pages),
            }

            if reader.metadata:
                if reader.metadata.title:
                    metadata["title"] = str(reader.metadata.title)
                if reader.metadata.author:
                    metadata["author"] = str(reader.metadata.author)

            logger.info(
                f"Successfully parsed PDF: {file_path.name} "
                f"({len(reader.pages)} pages, {len(full_content)} chars)"
            )

            return {
                "content": full_content,
                "elements": [],
                "element_types": ["Text"],
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path.name}: {str(e)}")
            raise ValueError(f"Failed to parse PDF: {str(e)}")


class TextParser:
    """
    Simple text file parser for TXT and Markdown files

    Handles plain text files with UTF-8 encoding.
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown"}

    @classmethod
    def is_supported(cls, file_extension: str) -> bool:
        """Check if file type is supported"""
        return file_extension.lower() in cls.SUPPORTED_EXTENSIONS

    async def parse(self, file_path: Path) -> Dict[str, Any]:
        """Parse text file"""
        try:
            import aiofiles

            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            if not content or len(content.strip()) < 10:
                raise ValueError("File appears to be empty or too short")

            file_type = file_path.suffix.lstrip('.').lower()

            logger.info(
                f"Successfully parsed {file_type.upper()}: {file_path.name} "
                f"({len(content)} chars)"
            )

            return {
                "content": content.strip(),
                "elements": [],
                "element_types": ["Text"],
                "metadata": {
                    "filename": file_path.name,
                    "filetype": file_type,
                    "page_count": 1,
                },
            }

        except UnicodeDecodeError:
            raise ValueError(
                f"Failed to decode file as UTF-8. File may be corrupted or in a different encoding."
            )
        except Exception as e:
            logger.error(f"Failed to parse text file {file_path.name}: {str(e)}")
            raise ValueError(f"Failed to parse text file: {str(e)}")


class DocumentParserFactory:
    """
    Factory for selecting appropriate document parser

    Supported formats:
    - PDF: .pdf (text-based PDFs only, no OCR)
    - Text: .txt
    - Markdown: .md, .markdown
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}

    @staticmethod
    def get_parser(file_extension: str):
        """
        Get appropriate parser for file type

        Args:
            file_extension: File extension (e.g., '.pdf', '.txt', '.md')

        Returns:
            Parser instance

        Raises:
            ValueError: If file type not supported
        """
        file_ext = file_extension.lower()

        if PDFParser.is_supported(file_ext):
            return PDFParser()

        if TextParser.is_supported(file_ext):
            return TextParser()

        raise ValueError(
            f"Unsupported file type: {file_ext}. "
            f"Supported extensions: {', '.join(sorted(DocumentParserFactory.SUPPORTED_EXTENSIONS))}"
        )

    @staticmethod
    def supported_extensions() -> set[str]:
        """Get all supported file extensions"""
        return DocumentParserFactory.SUPPORTED_EXTENSIONS
