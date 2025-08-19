"""
Modelos de datos
"""
from .document import (
    Document, 
    DocumentCollection, 
    TextChunk, 
    DocumentSummary,
    DocumentMetadata,
    DocumentStatus
)
from .conversation import (
    ConversationSession, 
    Message, 
    MessageRole, 
    ConversationContext
)

__all__ = [
    "Document", "DocumentCollection", "TextChunk", 
    "DocumentSummary", "DocumentMetadata", "DocumentStatus",
    "ConversationSession", "Message", "MessageRole", 
    "ConversationContext"
]