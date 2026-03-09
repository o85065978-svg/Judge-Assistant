"""
file_ingestor.py

Unified file ingestion service that handles text, PDF, and image files.

Workflow per file:
1. Detect file type (text / PDF / image).
2. Extract text:
   - Text files (.txt, .text, .csv, .json, .md) -> read directly.
   - PDF files (.pdf) -> extract text with PyPDF2.
   - Image files (.png, .jpg, .jpeg, .tiff, .bmp, .webp) -> run OCR.
3. Classify the document using the document classifier.
4. Store the document record in MongoDB.
5. Index the document text in the Chroma vector store so the Case Doc RAG
   can retrieve it dynamically.

This service can be called:
- **Before** a case run, to pre-load documents into the system.
- **During** a run, from the classify_and_store_document node.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from pymongo import MongoClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File type constants
# ---------------------------------------------------------------------------

TEXT_EXTENSIONS = {".txt", ".text", ".csv", ".json", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Lazy imports to avoid circular dependencies
# ---------------------------------------------------------------------------

def _get_classifier():
    """Lazy-import the document classifier."""
    classifier_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "RAG", "Case Doc RAG",
    )
    classifier_dir = os.path.normpath(classifier_dir)
    if classifier_dir not in sys.path:
        sys.path.insert(0, classifier_dir)

    from document_classifier import classify_document
    return classify_document


def _get_ocr_processor():
    """Lazy-import the OCR pipeline's process_document function."""
    ocr_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "OCR",
    )
    ocr_dir = os.path.normpath(ocr_dir)
    if ocr_dir not in sys.path:
        sys.path.insert(0, ocr_dir)

    from ocr_pipeline import process_document
    return process_document


# ---------------------------------------------------------------------------
# File type detection
# ---------------------------------------------------------------------------

def detect_file_type(file_path: str) -> str:
    """Return ``'text'``, ``'pdf'``, ``'image'``, or ``'unknown'``."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext in TEXT_EXTENSIONS:
        return "text"
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def extract_text_from_file(file_path: str) -> str:
    """Read a plain text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2.

    Falls back to an empty string if extraction fails.
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.error(
            "PyPDF2 is required for PDF text extraction. "
            "Install it with: pip install PyPDF2"
        )
        return ""

    try:
        reader = PdfReader(file_path)
        pages_text: List[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        return "\n\n".join(pages_text)
    except Exception as exc:
        logger.exception("Failed to extract text from PDF '%s': %s", file_path, exc)
        return ""


def extract_text_via_ocr(file_path: str, doc_id: Optional[str] = None) -> str:
    """Run the OCR pipeline on an image file and return extracted text."""
    try:
        process_document = _get_ocr_processor()
        result = process_document(file_path=file_path, doc_id=doc_id)
        return result.raw_text
    except Exception as exc:
        logger.exception("OCR failed for '%s': %s", file_path, exc)
        return ""


# ---------------------------------------------------------------------------
# FileIngestor - main service class
# ---------------------------------------------------------------------------

class FileIngestor:
    """Handles end-to-end ingestion of files into MongoDB and the vector store.

    Parameters
    ----------
    mongo_uri : str
        MongoDB connection URI.
    mongo_db : str
        Database name (default ``"Rag"``).
    mongo_collection : str
        Collection name (default ``"Document Storage"``).
    embedding_model : str
        HuggingFace embedding model for the vector store.
    chroma_collection : str
        Chroma collection name shared with the Case Doc RAG.
    chroma_persist_dir : str or None
        If provided, Chroma will persist to this directory.
    """

    def __init__(
        self,
        mongo_uri: Optional[str] = None,
        mongo_db: str = "Rag",
        mongo_collection: str = "Document Storage",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        chroma_collection: str = "judicial_docs",
        chroma_persist_dir: Optional[str] = None,
    ):
        self._mongo_uri = mongo_uri or os.getenv(
            "MONGO_URI", "mongodb://localhost:27017/"
        )
        self._mongo_db_name = mongo_db
        self._mongo_col_name = mongo_collection
        self._embedding_model_name = embedding_model
        self._chroma_collection_name = chroma_collection
        self._chroma_persist_dir = chroma_persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR", ""
        )

        # Lazily initialised
        self._mongo_client: Optional[MongoClient] = None
        self._vectorstore = None
        self._classifier = None

    # -- Lazy accessors ---------------------------------------------------

    @property
    def mongo_collection(self):
        """Return the MongoDB collection, connecting if needed."""
        if self._mongo_client is None:
            self._mongo_client = MongoClient(self._mongo_uri)
        db = self._mongo_client[self._mongo_db_name]
        return db[self._mongo_col_name]

    @property
    def vectorstore(self):
        """Return the Chroma vector store, creating if needed."""
        if self._vectorstore is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma

            embeddings = HuggingFaceEmbeddings(
                model_name=self._embedding_model_name,
            )

            kwargs = {
                "collection_name": self._chroma_collection_name,
                "embedding_function": embeddings,
            }
            if self._chroma_persist_dir:
                kwargs["persist_directory"] = self._chroma_persist_dir

            self._vectorstore = Chroma(**kwargs)
        return self._vectorstore

    @property
    def classifier(self):
        """Return the document classifier function."""
        if self._classifier is None:
            self._classifier = _get_classifier()
        return self._classifier

    # -- Core ingestion ---------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        case_id: str = "",
        pre_extracted_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a single file: extract text, classify, store in MongoDB,
        and index in the vector store.

        Parameters
        ----------
        file_path : str
            Path to the file on disk.
        case_id : str
            The case this document belongs to.
        pre_extracted_text : str or None
            If provided, skip text extraction and use this text directly.
            Useful when OCR has already been run by another node.

        Returns
        -------
        dict
            Classification and storage result with keys:
            ``file``, ``title``, ``doc_type``, ``confidence``,
            ``explanation``, ``mongo_id``, ``file_type``.
        """
        file_type = detect_file_type(file_path) if not pre_extracted_text else "pre_extracted"

        # 1. Extract text
        text = pre_extracted_text or self._extract_text(file_path, file_type, case_id)

        if not text or not text.strip():
            logger.warning("No text extracted from '%s'", file_path)
            return {
                "file": file_path,
                "title": "",
                "doc_type": "unknown",
                "confidence": 0,
                "explanation": "No text could be extracted",
                "mongo_id": None,
                "file_type": file_type,
            }

        # 2. Classify
        classification = self.classifier(text)
        doc_type = classification.get("final_type", "مستند غير معروف")
        confidence = classification.get("confidence", 0)
        explanation = classification.get("explanation", "")
        title = doc_type

        # 3. Store in MongoDB
        mongo_id = self._store_in_mongo(
            title=title,
            doc_type=doc_type,
            case_id=case_id,
            source_file=file_path,
            text=text,
            confidence=confidence,
            explanation=explanation,
            file_type=file_type,
        )

        # 4. Index in vector store
        self._index_in_vectorstore(
            text=text,
            title=title,
            doc_type=doc_type,
            case_id=case_id,
            source_file=file_path,
            mongo_id=str(mongo_id) if mongo_id else "",
        )

        logger.info(
            "Ingested '%s': type='%s', confidence=%d, mongo_id=%s",
            file_path, doc_type, confidence, mongo_id,
        )

        return {
            "file": file_path,
            "title": title,
            "doc_type": doc_type,
            "confidence": confidence,
            "explanation": explanation,
            "mongo_id": str(mongo_id) if mongo_id else None,
            "file_type": file_type,
        }

    def ingest_files(
        self,
        file_paths: List[str],
        case_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Ingest multiple files. Convenience wrapper around ``ingest_file``.

        Parameters
        ----------
        file_paths : list of str
            Paths to files on disk.
        case_id : str
            The case these documents belong to.

        Returns
        -------
        list of dict
            One result dict per file.
        """
        results = []
        for fp in file_paths:
            try:
                result = self.ingest_file(fp, case_id=case_id)
                results.append(result)
            except Exception as exc:
                logger.exception("Failed to ingest '%s': %s", fp, exc)
                results.append({
                    "file": fp,
                    "title": "",
                    "doc_type": "unknown",
                    "confidence": 0,
                    "explanation": f"Ingestion failed: {exc}",
                    "mongo_id": None,
                    "file_type": detect_file_type(fp),
                })
        return results

    def ingest_ocr_results(
        self,
        raw_texts: List[str],
        uploaded_files: List[str],
        case_id: str = "",
    ) -> List[Dict[str, Any]]:
        """Ingest pre-extracted OCR texts (already processed by the OCR adapter).

        Parameters
        ----------
        raw_texts : list of str
            Text strings extracted by the OCR pipeline.
        uploaded_files : list of str
            Corresponding file paths for reference.
        case_id : str
            The case these documents belong to.

        Returns
        -------
        list of dict
            One result dict per text.
        """
        results = []
        for i, text in enumerate(raw_texts):
            file_ref = uploaded_files[i] if i < len(uploaded_files) else f"ocr_doc_{i}"
            if not text or not text.strip():
                logger.warning("Skipping empty OCR text at index %d", i)
                continue
            try:
                result = self.ingest_file(
                    file_path=file_ref,
                    case_id=case_id,
                    pre_extracted_text=text,
                )
                results.append(result)
            except Exception as exc:
                logger.exception("Failed to ingest OCR text %d: %s", i, exc)
                results.append({
                    "file": file_ref,
                    "title": "",
                    "doc_type": "unknown",
                    "confidence": 0,
                    "explanation": f"Ingestion failed: {exc}",
                    "mongo_id": None,
                    "file_type": "image",
                })
        return results

    # -- Internal helpers -------------------------------------------------

    def _extract_text(
        self, file_path: str, file_type: str, case_id: str
    ) -> str:
        """Extract text from a file based on its detected type."""
        if file_type == "text":
            return extract_text_from_file(file_path)
        elif file_type == "pdf":
            return extract_text_from_pdf(file_path)
        elif file_type == "image":
            return extract_text_via_ocr(file_path, doc_id=case_id)
        else:
            logger.warning(
                "Unknown file type for '%s'. Attempting text read.", file_path
            )
            try:
                return extract_text_from_file(file_path)
            except Exception:
                return ""

    def _store_in_mongo(
        self,
        title: str,
        doc_type: str,
        case_id: str,
        source_file: str,
        text: str,
        confidence: int,
        explanation: str,
        file_type: str,
    ) -> Optional[str]:
        """Insert a document record into MongoDB. Returns the inserted ID."""
        doc_record = {
            "title": title,
            "doc_type": doc_type,
            "case_id": case_id,
            "source_file": source_file,
            "text": text,
            "classification_confidence": confidence,
            "classification_explanation": explanation,
            "file_type": file_type,
        }
        try:
            result = self.mongo_collection.insert_one(doc_record)
            logger.info(
                "Stored in MongoDB: title='%s', id=%s", title, result.inserted_id
            )
            return result.inserted_id
        except Exception as exc:
            logger.exception("MongoDB insert failed for '%s': %s", title, exc)
            return None

    def _index_in_vectorstore(
        self,
        text: str,
        title: str,
        doc_type: str,
        case_id: str,
        source_file: str,
        mongo_id: str,
    ) -> None:
        """Split the text into chunks and add them to the Chroma vector store.

        Each chunk carries metadata so the Case Doc RAG retriever can filter
        by case_id or doc_type.
        """
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        chunks = splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks produced from text for '%s'", source_file)
            return

        metadatas = [
            {
                "title": title,
                "type": doc_type,
                "case_id": case_id,
                "source_file": source_file,
                "mongo_id": mongo_id,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        try:
            self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            logger.info(
                "Indexed %d chunk(s) in vector store for '%s' (case=%s)",
                len(chunks), title, case_id,
            )
        except Exception as exc:
            logger.exception(
                "Vector store indexing failed for '%s': %s", source_file, exc
            )
