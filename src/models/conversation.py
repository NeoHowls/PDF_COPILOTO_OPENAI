
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    #Rol mensaje
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ConversationContext(BaseModel):
    #Contexto
    relevant_chunks: List[str] = Field(default_factory=list)
    mentioned_documents: List[str] = Field(default_factory=list)
    topics_discussed: List[str] = Field(default_factory=list)
    last_query_type: Optional[str] = None


class Message(BaseModel):
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    context: Optional[ConversationContext] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ConversationSession(BaseModel):
    id: str
    messages: List[Message] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime
    
    def add_message(self, message: Message):
        self.messages.append(message)
        self.updated_at = datetime.now()
        
    def get_conversation_history(self, limit: Optional[int] = None) -> List[Message]:
        #Historial
        messages = self.messages
        if limit:
            messages = messages[-limit:]
        return messages
        
    def get_context_summary(self) -> Dict[str, Any]:
        all_topics = set()
        all_docs = set()
        all_chunks = set()
        
        for msg in self.messages:
            if msg.context:
                all_topics.update(msg.context.topics_discussed)
                all_docs.update(msg.context.mentioned_documents)
                all_chunks.update(msg.context.relevant_chunks)
        
        return {
            "topics_discussed": list(all_topics),
            "documents_referenced": list(all_docs),
            "chunks_used": list(all_chunks),
            "message_count": len(self.messages)
        }