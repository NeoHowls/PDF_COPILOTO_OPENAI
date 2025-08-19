
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"


class DocumentMetadata(BaseModel):
    filename: str
    file_size: int
    page_count: int
    upload_time: datetime
    processing_time: Optional[float] = None
    language: Optional[str] = None
    
    
class TextChunk(BaseModel):
    id: str
    content: str
    page_number: int
    chunk_index: int
    document_id: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentSummary(BaseModel):
    document_id: str
    title: str
    main_topics: List[str]
    summary: str
    key_points: List[str]
    word_count: int
    

class Document(BaseModel):
    id: str
    filename: str
    status: DocumentStatus
    metadata: DocumentMetadata
    chunks: List[TextChunk] = Field(default_factory=list)
    summary: Optional[DocumentSummary] = None
    topics: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class DocumentCollection(BaseModel):
    documents: List[Document] = Field(default_factory=list)
    total_chunks: int = 0
    total_pages: int = 0
    
    def add_document(self, document: Document):
        self.documents.append(document)
        self.total_chunks += len(document.chunks)
        self.total_pages += document.metadata.page_count
        
    def get_document_by_id(self, doc_id: str) -> Optional[Document]:
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
        
    def get_all_chunks(self) -> List[TextChunk]:
        chunks = []
        for doc in self.documents:
            chunks.extend(doc.chunks)
        return chunks