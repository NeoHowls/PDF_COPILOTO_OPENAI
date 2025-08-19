
import pytest
from unittest.mock import Mock
from src.core.vector_store import VectorStore


class TestVectorStore:
    def test_init(self):
        # Mock ChromaDB para testing
        with pytest.raises(Exception):
            # Esperamos que falle sin ChromaDB real
            store = VectorStore()