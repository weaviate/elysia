# Current State: Elysia Backend (specter-backend)

**Date**: 2025-10-11 (Updated)
**Status**: Document Upload Feature - Production Ready

---

## 1. Architecture Overview

### Tech Stack
- **Framework**: FastAPI 0.115.11+
- **Vector Database**: Weaviate (Cloud & Local support)
- **Python**: 3.10-3.12
- **Key Dependencies**:
  - `weaviate-client>=4.16.7` - Weaviate integration
  - `python-multipart==0.0.18` - File upload support
  - `pypdf>=4.0.0` - PDF text extraction
  - `aiofiles>=23.2.1` - Async file operations
  - `spacy==3.8.7` - NLP and tokenization
  - `dspy-ai>=3.0.0` - LLM interactions
  - `litellm>=1.74.15` - Multi-provider LLM support

### Application Structure
```
specter-backend/
├── elysia/
│   ├── api/                    # FastAPI application
│   │   ├── routes/            # API endpoints
│   │   │   ├── documents.py   # Document upload/list/delete
│   │   │   └── collections.py # Collection management
│   │   ├── services/          # Business logic services
│   │   │   └── document.py    # Document processing service
│   │   ├── dependencies/      # Dependency injection
│   │   ├── middleware/        # Error handlers, etc.
│   │   ├── utils/             # Utilities
│   │   ├── api_types.py       # Pydantic models
│   │   └── app.py             # Main FastAPI app
│   ├── preprocessing/         # Collection preprocessing
│   ├── tools/                 # Built-in tools
│   │   └── retrieval/         # Retrieval & chunking
│   └── util/                  # Core utilities
│       ├── client.py          # Weaviate ClientManager
│       ├── collection.py      # Collection utilities
│       └── document_parser.py # PDF/TXT/MD parsers
└── docs/                      # Documentation
```

---

## 2. Existing Weaviate Integration

### ClientManager (elysia/util/client.py)
- **Purpose**: Manages Weaviate client connections
- **Key Features**:
  - Async and sync client support
  - Connection pooling with timeout management
  - Multiple API key support (OpenAI, Cohere, etc.)
  - Local and cloud Weaviate support
  - Thread-safe operations with locks

**Environment Configuration**:
```env
WCD_URL=spviqddpqfccjxccakcpq.c0.europe-west3.gcp.weaviate.cloud
WCD_API_KEY=K05w...
WEAVIATE_IS_LOCAL=False
OPENAI_API_KEY=sk-proj-...
BASE_MODEL=gpt-4o-mini
COMPLEX_MODEL=gpt-4o
```

### Current Collections
- Regular user collections (no ELYSIA_ prefix)
- `ELYSIA_METADATA__` - Stores preprocessed collection metadata
- `ELYSIA_CHUNKED_{collection_name}__` - Stores chunked documents
- `ELYSIA_UPLOADED_DOCUMENTS` - Stores uploaded documents (PDF, TXT, MD)

---

## 3. Existing API Routes

### Current Endpoints (elysia/api/routes/)

#### Document Routes (`documents.py`) ✅ NEW
- `POST /documents/{user_id}/upload` - Upload document (PDF, TXT, MD)
- `GET /documents/{user_id}/list` - List user's uploaded documents
- `DELETE /documents/{user_id}/delete/{document_id}` - Delete document and chunks
- `GET /documents/supported-formats` - Get supported file formats

#### Collections Routes (`collections.py`)
- `GET /collections/mapping_types` - Get available data type mappings
- `GET /collections/{user_id}/list` - List all collections
- `POST /collections/{user_id}/view/{collection_name}` - Paginated view
- `GET /collections/{user_id}/get_object/{collection_name}/{uuid}` - Get single object
- `GET /collections/{user_id}/metadata/{collection_name}` - Get preprocessed metadata
- `PATCH /collections/{user_id}/metadata/{collection_name}` - Update metadata
- `DELETE /collections/{user_id}/metadata/{collection_name}` - Delete metadata
- `DELETE /collections/{user_id}/metadata/delete/all` - Delete all metadata

#### Other Routes
- `init.py` - Initialization endpoints
- `query.py` - WebSocket query endpoints
- `processor.py` - WebSocket processing
- `feedback.py` - User feedback
- `user_config.py` - User configuration
- `tree_config.py` - Decision tree configuration
- `utils.py` - Utility endpoints
- `tools.py` - Tool management
- `db.py` - Database operations

### Dependencies
- **User Management**: `get_user_manager()` - Provides UserManager singleton
- **ClientManager**: Accessed via `user_local["client_manager"]`

---

## 4. Data Models (elysia/api/api_types.py)

### Existing Pydantic Models
All API types use Pydantic for validation:
- `QueryData` - Query requests
- `ViewPaginatedCollectionData` - Collection viewing
- `UpdateCollectionMetadataData` - Metadata updates
- `ProcessCollectionData` - Collection processing
- `AddFeedbackData`, `RemoveFeedbackData` - User feedback
- `SaveConfigUserData`, `SaveConfigTreeData` - Configuration

**Pattern**: All models are well-structured with type hints and optional fields.

---

## 5. Chunking & Preprocessing System

### Chunker Class (elysia/tools/retrieval/chunk.py)
**Purpose**: Splits documents into smaller chunks for embedding

**Strategies**:
1. **Sentence-based**: Chunks by N sentences with overlap
2. **Token-based**: Chunks by N tokens with overlap

**Features**:
- Uses SpaCy for tokenization and sentence detection
- Configurable chunk size and overlap
- Span annotation tracking (start/end character positions)
- UUID generation for chunks

### AsyncCollectionChunker
**Purpose**: Async chunking workflow for Weaviate collections

**Workflow**:
1. Creates `ELYSIA_CHUNKED_{collection}__` collection
2. Inherits vectorizer from original collection
3. Chunks documents in parallel
4. Stores chunks with references to original documents
5. Creates bidirectional references (full ↔ chunked)

**Key Methods**:
- `chunk_single_object()` - Chunk one document
- `chunk_objects_parallel()` - Batch chunking
- `insert_chunks()` - Batch insert to Weaviate
- `insert_references()` - Create document-chunk relationships

---

## 6. Preprocessing System (elysia/preprocessing/collection.py)

### preprocess_async()
**Purpose**: Analyze and prepare collections for Elysia

**Steps**:
1. Sample objects from collection
2. Generate AI summary of dataset
3. Evaluate field statistics (range, groups, mean, etc.)
4. Determine appropriate return types (generic, conversation, message, etc.)
5. Create field mappings (data → frontend display)
6. Store metadata in `ELYSIA_METADATA__` collection

**Key Features**:
- LLM-powered analysis using DSPy
- Configurable sampling (min/max size, token limits)
- Field statistics (numeric ranges, date ranges, text lengths, boolean percentages)
- Caching in Weaviate (skip if already processed)

---

## 7. Document Upload Feature ✅ IMPLEMENTED

### Overview
Complete document upload system with parsing, chunking, and storage in Weaviate. Uses simple, robust parsers instead of complex libraries.

### Architecture

#### Document Parsers (`elysia/util/document_parser.py`)
**Simple, production-ready parsers for common document types**

1. **PDFParser**
   - Uses `pypdf` library for text extraction
   - Text-based PDFs only (no OCR)
   - Async parsing with ThreadPoolExecutor
   - Extracts metadata (title, author, page count)
   - Validates content is not empty

2. **TextParser**
   - Native UTF-8 file reading with `aiofiles`
   - Supports `.txt`, `.md`, `.markdown` files
   - Async file operations
   - Character encoding validation

3. **DocumentParserFactory**
   - Selects appropriate parser based on file extension
   - Centralized supported extensions list
   - Validates file types before parsing

**Supported Formats**: PDF, TXT, Markdown (50MB max file size)

#### Document Service (`elysia/api/services/document.py`)
**Business logic layer orchestrating the upload workflow**

**Key Features**:
- Automatic collection creation (`ELYSIA_UPLOADED_DOCUMENTS`)
- Document chunking with sentence-based strategy (5 sentences per chunk)
- Bidirectional references between documents and chunks
- Graceful preprocessing (continues even if preprocessing fails)
- User-based authorization and isolation

**Upload Workflow**:
1. Parse document → extract content and metadata
2. Ensure collection exists (create if needed)
3. Store full document in Weaviate with RFC3339 timestamp
4. Chunk document using existing chunking infrastructure
5. Create chunked collection with references
6. Preprocess collection for AI analysis (optional, graceful failure)

**Collection Schema** (`ELYSIA_UPLOADED_DOCUMENTS`):
```python
Properties:
- document_id (TEXT) - UUID identifier
- filename (TEXT) - Original filename
- content (TEXT) - Full document content (vectorized)
- file_type (TEXT) - Extension (pdf, txt, md)
- file_size (INT) - File size in bytes
- user_id (TEXT) - Owner user ID
- upload_date (DATE) - RFC3339 timestamp
- content_preview (TEXT) - First 500 characters
- element_types (TEXT_ARRAY) - Content types found
- metadata (OBJECT) - Nested metadata
  - title, author, filename, filetype, page_count

Vectorizer: text2vec-openai (text-embedding-3-small)
```

#### API Routes (`elysia/api/routes/documents.py`)
**FastAPI endpoints with robust error handling**

**Validation**:
- File size limit (50MB)
- File type validation
- Empty file detection
- User authorization checks
- Graceful error responses (no crashes)

**Edge Cases Handled**:
- Delete non-existent document → returns structured error
- Delete already-deleted document → returns "not found" message
- Unauthorized deletion attempts → returns 403 error
- Collection doesn't exist → graceful handling with empty results
- Preprocessing failures → document still uploads successfully
- Invalid file formats → clear error message with supported types

### Key Implementation Details

**RFC3339 Date Format**:
```python
datetime.now(UTC).isoformat().replace('+00:00', 'Z')
```

**Chunking Reference Creation**:
```python
await collection_chunker.create_chunked_reference(
    content_field="content",
    client_manager=self.client_manager
)
```

**Graceful Preprocessing**:
```python
if auto_preprocess:
    try:
        await self._preprocess_collection()
    except Exception as preprocess_error:
        logger.warning(f"Preprocessing failed (document still uploaded): {str(preprocess_error)}")
```

**DateTime JSON Serialization**:
```python
if isinstance(upload_date, datetime):
    upload_date = upload_date.isoformat()
```

---

## 8. User Workflows

### Existing Collections Workflow
**For user-created Weaviate collections:**
1. User manually creates collection in Weaviate (external)
2. User connects Elysia to their Weaviate cluster
3. Elysia lists available collections via API
4. User clicks "analyze" to preprocess collection
5. Collection is ready for querying

### Document Upload Workflow ✅ IMPLEMENTED
**For uploaded documents (PDF, TXT, Markdown):**
1. User uploads document via `POST /documents/{user_id}/upload`
2. Backend validates file (type, size, content)
3. Backend parses document using appropriate parser
4. Backend stores full document in `ELYSIA_UPLOADED_DOCUMENTS` collection
5. Backend chunks document into 5-sentence chunks
6. Backend creates `ELYSIA_CHUNKED_ELYSIA_UPLOADED_DOCUMENTS__` with references
7. Backend automatically preprocesses collection (optional, graceful failure)
8. Document is ready for semantic search and querying
9. User can list documents via `GET /documents/{user_id}/list`
10. User can delete documents via `DELETE /documents/{user_id}/delete/{document_id}`

**Response Example**:
```json
{
  "success": true,
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "collection_name": "ELYSIA_UPLOADED_DOCUMENTS",
  "filename": "report.pdf",
  "file_type": "pdf",
  "chunks_created": 42,
  "element_types": ["Text"],
  "message": "Document 'report.pdf' uploaded successfully"
}
```

---

## 9. Technical Implementation Details

### Async Architecture
- All file operations use `async`/`await` patterns
- PDF parsing offloaded to ThreadPoolExecutor (CPU-bound)
- Text parsing uses `aiofiles` for async I/O
- Weaviate operations use async client throughout
- Prevents blocking during file processing

### Resource Management
- **Temporary Files**: Created with `tempfile.NamedTemporaryFile`, deleted in `finally` block
- **Max File Size**: 50MB limit enforced before processing
- **Connection Pooling**: ClientManager handles Weaviate connection lifecycle
- **Memory Efficiency**: Streaming file reads, immediate cleanup

### Vector Embeddings
- **Vectorizer**: `text2vec-openai` with `text-embedding-3-small` model
- **Vectorized Field**: `content` (full document text)
- **Chunks**: Inherit vectorizer from parent collection
- Embeddings generated automatically by Weaviate on insert

### Error Handling Philosophy
- **Graceful Degradation**: Upload succeeds even if preprocessing fails
- **Structured Errors**: Always return JSON with `success`, `error`, `message` fields
- **No Crashes**: All edge cases handled with appropriate HTTP status codes
- **User-Friendly Messages**: Clear error descriptions with actionable information

### Date/Time Handling
- **Format**: RFC3339 with 'Z' timezone (`2025-10-11T15:09:21.643407Z`)
- **Timezone**: UTC (`datetime.now(UTC)`)
- **Serialization**: Convert to ISO string before JSON response
- **Storage**: Weaviate DATE type with proper RFC3339 validation

### Logging Configuration
- **Framework**: Python logging with Rich handler
- **Markup**: Disabled (`markup=False`) to prevent parsing issues with PDF metadata
- **Suppressed Loggers**: `pdfminer`, `PIL` (debug noise reduction)
- **Levels**: INFO for successful operations, WARNING for graceful failures, ERROR for exceptions

---

## 10. Integration Architecture

### Component Integration Map

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                      │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │        Routes (elysia/api/routes/)                  │   │
│  │                                                     │   │
│  │  documents.py ──────► DocumentService              │   │
│  │       │                     │                      │   │
│  │       │                     ├──► DocumentParser    │   │
│  │       │                     │        ├─► PDFParser│   │
│  │       │                     │        └─► TextParser  │   │
│  │       │                     │                      │   │
│  │       │                     ├──► Chunker           │   │
│  │       │                     ├──► AsyncCollectionChunker│
│  │       │                     └──► preprocess_async  │   │
│  │       │                                            │   │
│  │       └──► get_user_manager() ──► ClientManager   │   │
│  │                                                     │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│              ┌─────────────────────────┐                    │
│              │   Weaviate Cluster      │                    │
│              │                         │                    │
│              │  ELYSIA_UPLOADED_DOCUMENTS│                  │
│              │  ELYSIA_CHUNKED_*       │                    │
│              │  ELYSIA_METADATA__      │                    │
│              └─────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Integration Points

1. **Route → Service Layer**
   - `documents.py` routes delegate to `DocumentService`
   - Follows existing pattern from `collections.py`
   - Uses `get_user_manager()` dependency for user isolation

2. **Service → Parsers**
   - `DocumentService` uses `DocumentParserFactory.get_parser()`
   - Parser selection based on file extension
   - Async parsing for non-blocking operations

3. **Service → Chunking**
   - Reuses existing `Chunker` class (sentence-based, 5 sentences)
   - Reuses existing `AsyncCollectionChunker` for Weaviate storage
   - Creates bidirectional references (`isChunked`, `isFullOf`)

4. **Service → Preprocessing**
   - Calls existing `preprocess_async()` after upload
   - Generates AI summary and field mappings
   - Stores metadata in `ELYSIA_METADATA__`

5. **Service → Weaviate**
   - Uses `ClientManager` from user context
   - Async client for all operations
   - Automatic collection creation with proper schema

---

## 11. Summary

### Current State
The Elysia backend now has a **production-ready document upload feature** that allows users to upload PDF, TXT, and Markdown files. Documents are automatically parsed, chunked, vectorized, and made searchable.

### System Strengths
- ✅ **Simple, Robust Parsers**: Uses pypdf and native file reading (no complex dependencies)
- ✅ **Production-Ready Chunking**: Leverages existing sentence-based chunking infrastructure
- ✅ **Automatic Preprocessing**: AI-powered analysis and metadata generation
- ✅ **Graceful Error Handling**: No crashes, all edge cases handled with structured errors
- ✅ **User Isolation**: Documents scoped to user_id, authorization enforced
- ✅ **Async Throughout**: Non-blocking file operations and database calls
- ✅ **Bidirectional References**: Documents ↔ chunks for efficient retrieval

### Design Decisions
1. **Simplicity over Features**: Chose pypdf over Unstructured library
   - Reason: Unstructured had certificate issues, large dependencies, unreliable parsing
   - Trade-off: No OCR support, but much more reliable for text-based documents

2. **Graceful Degradation**: Upload succeeds even if preprocessing fails
   - Reason: Preprocessing requires valid OpenAI API key, shouldn't block upload
   - Result: Documents are still searchable, preprocessing can be retried later

3. **Single Document Collection**: All uploaded documents go to `ELYSIA_UPLOADED_DOCUMENTS`
   - Reason: Simpler management, user_id provides isolation
   - Benefit: Consistent schema, easier to query across all documents

4. **Sentence-Based Chunking**: Fixed 5-sentence chunks with overlap
   - Reason: Works well for most document types, proven in existing system
   - Benefit: Consistent chunk sizes, good for semantic search

### Known Limitations
- **PDF**: Text-based PDFs only (no OCR for scanned documents)
- **File Size**: 50MB max (configurable)
- **File Types**: PDF, TXT, Markdown only (no Word, Excel, PowerPoint)
- **Preprocessing**: Requires valid OpenAI API key (gracefully skipped if missing)

### Future Enhancements (Not Implemented)
- Batch upload support (multiple files at once)
- Progress tracking for large files
- OCR support for scanned PDFs
- Word document support (.docx)
- File versioning (upload new version of same document)
- Document tagging/categorization
- Full-text search across all user documents
