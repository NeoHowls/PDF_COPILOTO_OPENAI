
import pytest
import tempfile
from src.core.document_processor import DocumentProcessor


class TestDocumentProcessor:
    def test_init(self):
        processor = DocumentProcessor()
        assert processor is not None
    
    def test_validate_pdf(self):
        processor = DocumentProcessor()
        # Test con archivo inexistente
        result = processor.validate_pdf("nonexistent.pdf")
        assert not result["is_valid"]