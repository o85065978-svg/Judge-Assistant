"""
classify_and_store_document.py

Supervisor node that classifies uploaded documents (after OCR or directly)
and stores them in both MongoDB and the Chroma vector store.

This node runs after dispatch_agents when OCR was involved, or processes
raw text/PDF documents directly. It ensures every document stored in the
database has a proper classified name so the document selector in
rag_docs.py can find them reliably.

Supported file types:
- Text files (.txt, .text, .csv, .json, .md) -- read directly.
- PDF files (.pdf) -- text extracted with PyPDF2.
- Image files (.png, .jpg, .jpeg, .tiff, .bmp, .webp) -- processed via OCR.

After classification and MongoDB storage, document chunks are indexed in
the shared Chroma vector store so the Case Doc RAG can retrieve them
dynamically without a restart.
"""

import logging
from typing import Any, Dict, List

from Supervisor.config import (
    CHROMA_COLLECTION,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL,
    MONGO_COLLECTION,
    MONGO_DB,
    MONGO_URI,
)
from Supervisor.services.file_ingestor import FileIngestor, detect_file_type
from Supervisor.state import SupervisorState

logger = logging.getLogger(__name__)

# Module-level ingestor instance (reused across calls)
_ingestor = None


def _get_ingestor() -> FileIngestor:
    """Return the shared FileIngestor singleton."""
    global _ingestor
    if _ingestor is None:
        _ingestor = FileIngestor(
            mongo_uri=MONGO_URI,
            mongo_db=MONGO_DB,
            mongo_collection=MONGO_COLLECTION,
            embedding_model=EMBEDDING_MODEL,
            chroma_collection=CHROMA_COLLECTION,
            chroma_persist_dir=CHROMA_PERSIST_DIR,
        )
    return _ingestor


def classify_and_store_document_node(state: SupervisorState) -> Dict[str, Any]:
    """Classify each document from OCR output (or raw uploads) and store it.

    This node inspects ``agent_results`` for OCR output. If OCR was run,
    it takes the extracted text, classifies the document type, and stores
    it in MongoDB **and** the Chroma vector store.

    If no OCR was needed but files were uploaded directly (text or PDF),
    it reads/extracts text, classifies, and stores in both backends.

    Updates state keys: ``document_classifications``.
    """
    agent_results = state.get("agent_results", {})
    uploaded_files = state.get("uploaded_files", [])
    case_id = state.get("case_id", "")

    ingestor = _get_ingestor()
    classifications: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Case 1: OCR was run -- ingest the pre-extracted texts
    # -----------------------------------------------------------------
    ocr_result = agent_results.get("ocr")
    if ocr_result:
        raw_texts = ocr_result.get("raw_output", {}).get("raw_texts", [])
        if not raw_texts:
            combined = ocr_result.get("response", "")
            if combined:
                raw_texts = [combined]

        results = ingestor.ingest_ocr_results(
            raw_texts=raw_texts,
            uploaded_files=uploaded_files,
            case_id=case_id,
        )
        classifications.extend(results)

        logger.info(
            "Classified and stored %d document(s) from OCR output",
            len(results),
        )

    # -----------------------------------------------------------------
    # Case 2: No OCR, but files were uploaded directly (text, PDF, etc.)
    # -----------------------------------------------------------------
    elif uploaded_files:
        for file_path in uploaded_files:
            file_type = detect_file_type(file_path)

            if file_type == "unknown":
                logger.info(
                    "Skipping unsupported file type: %s", file_path,
                )
                continue

            # Images without prior OCR -- the ingestor will route
            # them through OCR automatically.
            try:
                result = ingestor.ingest_file(
                    file_path=file_path,
                    case_id=case_id,
                )
                classifications.append(result)
            except Exception as exc:
                logger.exception(
                    "Failed to ingest file '%s': %s", file_path, exc,
                )

        logger.info(
            "Classified and stored %d document(s) from direct uploads",
            len(classifications),
        )

    return {
        "document_classifications": classifications,
    }
