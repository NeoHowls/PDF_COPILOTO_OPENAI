
import pytest
from unittest.mock import Mock
from src.core.orchestrator import ConversationOrchestrator


class TestConversationOrchestrator:
    def test_init(self):
        mock_vector_store = Mock()
        mock_llm_service = Mock()
        
        orchestrator = ConversationOrchestrator(
            vector_store=mock_vector_store,
            llm_service=mock_llm_service
        )
        
        assert orchestrator.vector_store == mock_vector_store
        assert orchestrator.llm_service == mock_llm_service