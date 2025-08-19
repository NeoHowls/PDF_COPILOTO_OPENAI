
#Configuración centralizada del sistema

import os
from typing import Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
     #Configuración de la aplicación 
    
    # API Keys OPENAI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    # API Keys CLAUDE , -NO PROBADA pero debia funcionar
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    
    # Configuración de archivos
    max_pdfs: int = Field(5, env="MAX_PDFS")
    max_file_size_mb: int = Field(10, env="MAX_FILE_SIZE_MB")
    
    # Configuración de procesamiento de texto
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # Configuración de ChromaDB
    chroma_host: str = Field("chromadb", env="CHROMA_HOST")
    chroma_port: int = Field(8000, env="CHROMA_PORT")
    
    # Configuración de embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(384, env="EMBEDDING_DIMENSIONS")
    
    # Configuración de logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Directorios
    uploads_dir: str = "data/uploads"
    vectorstore_dir: str = "data/vectorstore"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()