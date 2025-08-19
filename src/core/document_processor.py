
#Procesador de documentos PDF

import os
import uuid
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument

from models.document import (
    Document, DocumentMetadata, TextChunk, DocumentStatus, DocumentSummary
)
from config.settings import settings
from utils.pdf_utils import extract_text_from_pdf, detect_language
from utils.text_utils import clean_text, extract_keywords


class DocumentProcessor:
 # Procesar PDF
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def process_pdf(self, file_path: str, filename: str) -> Document:
        start_time = time.time()
        
        try:
            # Extraer texto del PDF
            text_content, page_count = extract_text_from_pdf(file_path)
            
            if not text_content.strip():
                raise ValueError("No se pudo extraer texto del PDF")
            
            # Crear metadatos
            file_size = os.path.getsize(file_path)
            metadata = DocumentMetadata(
                filename=filename,
                file_size=file_size,
                page_count=page_count,
                upload_time=datetime.now(),
                language=detect_language(text_content[:1000])  # Detectar idioma con muestra
            )
            
            # Crear documento
            doc_id = str(uuid.uuid4())
            document = Document(
                id=doc_id,
                filename=filename,
                status=DocumentStatus.PROCESSING,
                metadata=metadata
            )
            
            # Dividir texto en chunks
            chunks = self._create_chunks(text_content, doc_id)
            document.chunks = chunks
            
            # Extraer temas principales
            document.topics = self._extract_topics(text_content)
            
            # Crear resumen
            document.summary = self._create_summary(text_content, doc_id, filename)
            
            # Actualizar estado y tiempo de procesamiento
            processing_time = time.time() - start_time
            document.metadata.processing_time = processing_time
            document.status = DocumentStatus.PROCESSED
            
            return document
            
        except Exception as e:
            # Crear documento con error
            doc_id = str(uuid.uuid4())
            error_metadata = DocumentMetadata(
                filename=filename,
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                page_count=0,
                upload_time=datetime.now()
            )
            
            document = Document(
                id=doc_id,
                filename=filename,
                status=DocumentStatus.ERROR,
                metadata=error_metadata
            )
            
            raise e
    
    def _create_chunks(self, text: str, document_id: str) -> List[TextChunk]:
        #texto
        chunks = []
        
        # Dividir texto en chunks
        text_chunks = self.text_splitter.split_text(clean_text(text))
        
        for i, chunk_content in enumerate(text_chunks):
            chunk = TextChunk(
                id=f"{document_id}_chunk_{i}",
                content=chunk_content,
                page_number=self._estimate_page_number(i, len(text_chunks)),
                chunk_index=i,
                document_id=document_id,
                metadata={
                    "word_count": len(chunk_content.split()),
                    "char_count": len(chunk_content)
                }
            )
            chunks.append(chunk)
            
        return chunks
    
    def _estimate_page_number(self, chunk_index: int, total_chunks: int) -> int:
        # Estimación simple basada en distribución uniforme
        return max(1, int((chunk_index / total_chunks) * 10) + 1)
    
    def _extract_topics(self, text: str) -> List[str]:
        # Implementación simple usando palabras clave
        keywords = extract_keywords(text, max_keywords=10)
        return keywords
    
    def _create_summary(self, text: str, doc_id: str, filename: str) -> DocumentSummary:
        # Tomar primeros párrafos para el resumen
        paragraphs = text.split('\n\n')
        summary_text = ' '.join(paragraphs[:3])[:500] + "..."
        
        # Extraer puntos clave (implementación simple)
        key_points = []
        for paragraph in paragraphs[:5]:
            if len(paragraph.strip()) > 100:
                key_points.append(paragraph.strip()[:200] + "...")
                if len(key_points) >= 3:
                    break
        
        return DocumentSummary(
            document_id=doc_id,
            title=filename.replace('.pdf', '').replace('_', ' ').title(),
            main_topics=self._extract_topics(text)[:5],
            summary=summary_text,
            key_points=key_points,
            word_count=len(text.split())
        )
    
    def batch_process_pdfs(self, file_paths: List[str]) -> List[Document]:
        documents = []
        #lote pdf
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                document = self.process_pdf(file_path, filename)
                documents.append(document)
            except Exception as e:
                print(f"Error procesando {file_path}: {str(e)}")
                continue
                
        return documents
    
    def validate_pdf(self, file_path: str) -> Dict[str, Any]:
        #validar
        validation_result = {
            "is_valid": False,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Verificar tamaño del archivo
            file_size = os.path.getsize(file_path)
            max_size_bytes = settings.max_file_size_mb * 1024 * 1024
            
            if file_size > max_size_bytes:
                validation_result["errors"].append(
                    f"Archivo muy grande: {file_size / (1024*1024):.1f}MB. Máximo: {settings.max_file_size_mb}MB"
                )
            
            # Verificar que es un PDF válido
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                if page_count == 0:
                    validation_result["errors"].append("El PDF no tiene páginas")
                elif page_count > 100:
                    validation_result["warnings"].append(f"PDF muy largo: {page_count} páginas")
                
                # Intentar extraer algo de texto
                text_sample = ""
                for i in range(min(3, page_count)):
                    try:
                        text_sample += pdf_reader.pages[i].extract_text()
                    except:
                        continue
                
                if not text_sample.strip():
                    validation_result["warnings"].append("PDF podría ser solo imágenes (sin texto extraíble)")
            
            validation_result["is_valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            validation_result["errors"].append(f"Error validando PDF: {str(e)}")
            
        return validation_result