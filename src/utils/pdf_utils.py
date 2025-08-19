
import PyPDF2
from typing import Tuple, Optional
import langdetect
from langdetect import DetectorFactory


# Hacer determinístico el detector de idioma
DetectorFactory.seed = 0


def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """
    Extraer texto de un archivo PDF
    
    Returns:
        Tuple[str, int]: (texto_extraido, numero_de_paginas)
    """
    text_content = []
    page_count = 0
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            
            for page_num in range(page_count):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        text_content.append(page_text)
                        
                except Exception as e:
                    print(f"Error extrayendo página {page_num + 1}: {e}")
                    continue
        
        full_text = "\n\n".join(text_content)
        return full_text, page_count
        
    except Exception as e:
        raise Exception(f"Error procesando PDF: {str(e)}")


def detect_language(text: str) -> Optional[str]:
    """
    Detectar idioma del texto
    
    Args:
        text: Muestra de texto para detectar idioma
        
    Returns:
        str: Código de idioma (es, en, fr, etc.) o None si no se puede detectar
    """
    if not text.strip():
        return None
        
    try:
        # Limpiar texto para mejor detección
        clean_text = text.replace('\n', ' ').strip()
        
        if len(clean_text) < 10:
            return None
            
        language = langdetect.detect(clean_text)
        return language
        
    except Exception:
        return None


def validate_pdf_structure(file_path: str) -> dict:
    """
    Validar estructura del PDF
    
    Returns:
        dict: Información sobre la estructura del PDF
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            info = {
                "is_valid": True,
                "page_count": len(pdf_reader.pages),
                "is_encrypted": pdf_reader.is_encrypted,
                "metadata": {},
                "has_extractable_text": False,
                "total_chars": 0
            }
            
            # Obtener metadatos
            if pdf_reader.metadata:
                metadata = pdf_reader.metadata
                info["metadata"] = {
                    "title": getattr(metadata, 'title', None),
                    "author": getattr(metadata, 'author', None),
                    "subject": getattr(metadata, 'subject', None),
                    "creator": getattr(metadata, 'creator', None),
                    "producer": getattr(metadata, 'producer', None),
                    "creation_date": str(getattr(metadata, 'creation_date', None)),
                    "modification_date": str(getattr(metadata, 'modification_date', None))
                }
            
            # Verificar si hay texto extraíble
            sample_text = ""
            pages_with_text = 0
            
            for i, page in enumerate(pdf_reader.pages[:min(5, len(pdf_reader.pages))]):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        sample_text += page_text
                        pages_with_text += 1
                        info["total_chars"] += len(page_text)
                except:
                    continue
            
            info["has_extractable_text"] = len(sample_text.strip()) > 0
            info["pages_with_text"] = pages_with_text
            info["sample_text"] = sample_text[:500] if sample_text else ""
            
            return info
            
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
            "page_count": 0,
            "is_encrypted": False,
            "has_extractable_text": False
        }


def extract_pdf_metadata(file_path: str) -> dict:
    """
    Extraer solo metadatos del PDF
    
    Returns:
        dict: Metadatos del PDF
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            metadata = {
                "page_count": len(pdf_reader.pages),
                "is_encrypted": pdf_reader.is_encrypted
            }
            
            if pdf_reader.metadata:
                pdf_metadata = pdf_reader.metadata
                metadata.update({
                    "title": getattr(pdf_metadata, 'title', None),
                    "author": getattr(pdf_metadata, 'author', None),
                    "subject": getattr(pdf_metadata, 'subject', None),
                    "creator": getattr(pdf_metadata, 'creator', None),
                    "producer": getattr(pdf_metadata, 'producer', None),
                    "creation_date": str(getattr(pdf_metadata, 'creation_date', None)),
                    "modification_date": str(getattr(pdf_metadata, 'modification_date', None))
                })
            
            return metadata
            
    except Exception as e:
        return {"error": str(e)}


def is_pdf_readable(file_path: str) -> bool:
    """
    Verificar si el PDF es legible (no está dañado)
    
    Returns:
        bool: True si el PDF es legible
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Intentar leer la primera página
            if len(pdf_reader.pages) > 0:
                first_page = pdf_reader.pages[0]
                first_page.extract_text()
                return True
            
            return False
            
    except Exception:
        return False


def get_pdf_page_text(file_path: str, page_number: int) -> Optional[str]:
    """
    Extraer texto de una página específica
    
    Args:
        file_path: Ruta al archivo PDF
        page_number: Número de página (0-indexed)
        
    Returns:
        str: Texto de la página o None si hay error
    """
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            if page_number < 0 or page_number >= len(pdf_reader.pages):
                return None
                
            page = pdf_reader.pages[page_number]
            return page.extract_text()
            
    except Exception:
        return None


def estimate_reading_time(text: str, words_per_minute: int = 200) -> dict:
    """
    Estimar tiempo de lectura del texto
    
    Args:
        text: Texto a analizar
        words_per_minute: Velocidad de lectura promedio
        
    Returns:
        dict: Información sobre tiempo de lectura
    """
    if not text.strip():
        return {"words": 0, "minutes": 0, "seconds": 0}
    
    word_count = len(text.split())
    total_minutes = word_count / words_per_minute
    minutes = int(total_minutes)
    seconds = int((total_minutes - minutes) * 60)
    
    return {
        "words": word_count,
        "characters": len(text),
        "minutes": minutes,
        "seconds": seconds,
        "total_minutes": round(total_minutes, 1),
        "reading_speed": words_per_minute
    }