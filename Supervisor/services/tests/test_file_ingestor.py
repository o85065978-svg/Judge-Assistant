"""
test_file_ingestor.py

Unit tests for the FileIngestor service.

These tests mock external dependencies (MongoDB, Chroma, OCR, classifier)
to verify the ingestion logic in isolation.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from Supervisor.services.file_ingestor import (
    FileIngestor,
    detect_file_type,
    extract_text_from_file,
)


# ---------------------------------------------------------------------------
# detect_file_type
# ---------------------------------------------------------------------------

class TestDetectFileType:
    def test_text_files(self):
        assert detect_file_type("doc.txt") == "text"
        assert detect_file_type("doc.text") == "text"
        assert detect_file_type("data.csv") == "text"
        assert detect_file_type("data.json") == "text"
        assert detect_file_type("notes.md") == "text"

    def test_pdf_files(self):
        assert detect_file_type("document.pdf") == "pdf"
        assert detect_file_type("REPORT.PDF") == "pdf"

    def test_image_files(self):
        assert detect_file_type("scan.png") == "image"
        assert detect_file_type("photo.jpg") == "image"
        assert detect_file_type("photo.jpeg") == "image"
        assert detect_file_type("scan.tiff") == "image"
        assert detect_file_type("img.bmp") == "image"
        assert detect_file_type("pic.webp") == "image"

    def test_unknown_files(self):
        assert detect_file_type("archive.zip") == "unknown"
        assert detect_file_type("binary.exe") == "unknown"
        assert detect_file_type("noextension") == "unknown"


# ---------------------------------------------------------------------------
# extract_text_from_file
# ---------------------------------------------------------------------------

class TestExtractTextFromFile:
    def test_reads_text_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("Hello, world!")
            f.flush()
            path = f.name

        try:
            text = extract_text_from_file(path)
            assert text == "Hello, world!"
        finally:
            os.unlink(path)

    def test_reads_arabic_text(self):
        content = "صحيفة دعوى مقدمة من المدعي"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = f.name

        try:
            text = extract_text_from_file(path)
            assert text == content
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# FileIngestor
# ---------------------------------------------------------------------------

class TestFileIngestor:
    """Tests for the FileIngestor class with mocked backends."""

    @pytest.fixture
    def mock_ingestor(self):
        """Create a FileIngestor with mocked MongoDB and vectorstore."""
        ingestor = FileIngestor.__new__(FileIngestor)
        ingestor._mongo_uri = "mongodb://test:27017/"
        ingestor._mongo_db_name = "TestDB"
        ingestor._mongo_col_name = "TestCol"
        ingestor._embedding_model_name = "test-model"
        ingestor._chroma_collection_name = "test_collection"
        ingestor._chroma_persist_dir = ""
        ingestor._mongo_client = None
        ingestor._vectorstore = None
        ingestor._classifier = None

        # Mock MongoDB collection
        mock_collection = MagicMock()
        mock_insert_result = MagicMock()
        mock_insert_result.inserted_id = "mock_mongo_id_123"
        mock_collection.insert_one.return_value = mock_insert_result

        # Mock vectorstore
        mock_vs = MagicMock()

        # Mock classifier
        mock_classifier = MagicMock(return_value={
            "final_type": "صحيفة دعوى",
            "confidence": 85,
            "explanation": "Test classification",
        })

        # Patch the properties
        type(ingestor).mongo_collection = property(lambda self: mock_collection)
        type(ingestor).vectorstore = property(lambda self: mock_vs)
        type(ingestor).classifier = property(lambda self: mock_classifier)

        return ingestor, mock_collection, mock_vs, mock_classifier

    def test_ingest_text_file(self, mock_ingestor):
        ingestor, mock_col, mock_vs, mock_clf = mock_ingestor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("محتوى المستند القانوني")
            path = f.name

        try:
            result = ingestor.ingest_file(path, case_id="case_001")

            assert result["doc_type"] == "صحيفة دعوى"
            assert result["confidence"] == 85
            assert result["mongo_id"] == "mock_mongo_id_123"
            assert result["file_type"] == "text"

            # Verify MongoDB was called
            mock_col.insert_one.assert_called_once()
            doc = mock_col.insert_one.call_args[0][0]
            assert doc["case_id"] == "case_001"
            assert doc["text"] == "محتوى المستند القانوني"

            # Verify vectorstore was called
            mock_vs.add_texts.assert_called_once()
        finally:
            os.unlink(path)

    def test_ingest_pre_extracted_text(self, mock_ingestor):
        ingestor, mock_col, mock_vs, _ = mock_ingestor

        result = ingestor.ingest_file(
            file_path="ocr_output.png",
            case_id="case_002",
            pre_extracted_text="نص مستخرج من الصورة",
        )

        assert result["doc_type"] == "صحيفة دعوى"
        assert result["mongo_id"] is not None
        mock_col.insert_one.assert_called_once()
        mock_vs.add_texts.assert_called_once()

    def test_ingest_empty_text_returns_early(self, mock_ingestor):
        ingestor, mock_col, mock_vs, _ = mock_ingestor

        result = ingestor.ingest_file(
            file_path="empty.png",
            case_id="case_003",
            pre_extracted_text="   ",
        )

        assert result["doc_type"] == "unknown"
        assert result["mongo_id"] is None
        mock_col.insert_one.assert_not_called()
        mock_vs.add_texts.assert_not_called()

    def test_ingest_ocr_results(self, mock_ingestor):
        ingestor, mock_col, mock_vs, _ = mock_ingestor

        results = ingestor.ingest_ocr_results(
            raw_texts=["نص أول", "نص ثاني"],
            uploaded_files=["img1.png", "img2.png"],
            case_id="case_004",
        )

        assert len(results) == 2
        assert mock_col.insert_one.call_count == 2
        assert mock_vs.add_texts.call_count == 2

    def test_ingest_files_batch(self, mock_ingestor):
        ingestor, mock_col, mock_vs, _ = mock_ingestor

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f1:
            f1.write("مستند أول")
            path1 = f1.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f2:
            f2.write("مستند ثاني")
            path2 = f2.name

        try:
            results = ingestor.ingest_files(
                [path1, path2], case_id="case_005"
            )
            assert len(results) == 2
            assert all(r["mongo_id"] is not None for r in results)
        finally:
            os.unlink(path1)
            os.unlink(path2)
