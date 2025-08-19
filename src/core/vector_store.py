
import uuid
import os
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from sentence_transformers import SentenceTransformer
import logging
import time

from models.document import TextChunk, Document
from config.settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
     #Manejo del vector store con ChromaDB 0.6.1
    
    def __init__(self):
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.collection_name = "document_chunks"
        
        self._initialize_client()
        self._load_embedding_model()
        self._initialize_collection()
    
    def _initialize_client(self):

        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # ChromaDB 0.6.1 - Nueva API para HttpClient
                self.client = chromadb.HttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                    # En 0.6.1 ya no se usan ChromaSettings aquí
                )
                
                # Probar conexión - método actualizado
                try:
                    # En 0.6.1, heartbeat() puede no estar disponible
                    # Usar list_collections() como health check
                    collections = self.client.list_collections()
                    logger.info(f"✅ Conectado a ChromaDB en {settings.chroma_host}:{settings.chroma_port}")
                    logger.info(f"Colecciones existentes: {len(collections)}")
                    return
                except AttributeError:
                    # Fallback para verificar conexión
                    try:
                        self.client.get_version()
                        logger.info(f"✅ Conectado a ChromaDB en {settings.chroma_host}:{settings.chroma_port}")
                        return
                    except:
                        # Último fallback - intentar operación básica
                        test_collections = []
                        try:
                            test_collections = self.client.list_collections()
                        except:
                            pass
                        logger.info(f"✅ Conectado a ChromaDB (básico)")
                        return
                
            except Exception as e:
                logger.warning(f"Intento {attempt + 1}/{max_retries} falló: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Reintentando en {retry_delay} segundos...")
                    time.sleep(retry_delay)
                else:
                    logger.error("No se pudo conectar a ChromaDB después de todos los intentos")
                    # Fallback a cliente persistente local
                    logger.warning("Intentando fallback a cliente persistente local...")
                    try:
                        persist_dir = getattr(settings, 'vector_store_path', 'data/vectorstore')
                        os.makedirs(persist_dir, exist_ok=True)
                        self.client = chromadb.PersistentClient(path=persist_dir)
                        logger.info(f"✅ Usando cliente persistente local: {persist_dir}")
                        return
                    except Exception as fallback_error:
                        logger.error(f"Error con cliente persistente: {fallback_error}")
                        raise ConnectionError(f"No se pudo conectar a ChromaDB: {e}")
    
    def _load_embedding_model(self):
        try:
            model_name = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"✅ Modelo de embeddings cargado: {model_name}")
        except Exception as e:
            logger.error(f"Error cargando modelo de embeddings: {e}")
            # Fallback a modelo más simple
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Usando modelo fallback: all-MiniLM-L6-v2")
            except Exception as fallback_error:
                logger.error(f"Error con modelo fallback: {fallback_error}")
                raise
    
    def _initialize_collection(self):
        if not self.client:
            raise RuntimeError("Cliente no inicializado")
            
        try:
            # ChromaDB 0.6.1 - Usar get_or_create_collection directamente
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Colección de chunks de documentos PDF"}
            )
            logger.info(f"✅ Colección '{self.collection_name}' inicializada")
            
        except Exception as e:
            logger.error(f"Error inicializando colección: {e}")
            # Intentar crear la colección explícitamente
            try:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Colección de chunks de documentos PDF"}
                )
                logger.info(f"✅ Nueva colección '{self.collection_name}' creada")
            except Exception as create_error:
                # Intentar obtener colección existente
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"✅ Colección existente '{self.collection_name}' obtenida")
                except Exception as get_error:
                    logger.error(f"Error total inicializando colección: {get_error}")
                    raise
    
    def add_document(self, document: Document):
        if not document.chunks:
            logger.warning(f"Documento {document.filename} no tiene chunks")
            return
        
        if not self.collection:
            raise RuntimeError("Colección no inicializada")
        
        try:
            # Preparar datos para ChromaDB
            ids = []
            documents = []
            metadatas = []
            embeddings = []
            
            logger.info(f"Procesando {len(document.chunks)} chunks para {document.filename}")
            
            for i, chunk in enumerate(document.chunks):
                try:
                    # Generar embedding
                    embedding = self.embedding_model.encode(chunk.content).tolist()
                    
                    # Preparar metadatos - ChromaDB 0.6.1 es más estricto con tipos
                    metadata = {
                        "document_id": str(document.id),
                        "filename": document.filename,
                        "page_number": int(chunk.page_number),
                        "chunk_index": int(chunk.chunk_index),
                        "word_count": int(chunk.metadata.get("word_count", 0)),
                        "char_count": int(chunk.metadata.get("char_count", 0))
                    }
                    
                    # Generar ID único para evitar duplicados
                    chunk_id = f"{document.id}_{chunk.chunk_index}_{hash(chunk.content) % 10000}"
                    
                    ids.append(chunk_id)
                    documents.append(chunk.content)
                    metadatas.append(metadata)
                    embeddings.append(embedding)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Procesados {i + 1}/{len(document.chunks)} chunks")
                        
                except Exception as chunk_error:
                    logger.error(f"Error procesando chunk {i}: {chunk_error}")
                    continue
            
            if not ids:
                logger.error("No se pudo procesar ningún chunk")
                return
            
            # Agregar a ChromaDB en lotes
            batch_size = 50
            total_added = 0
            
            for i in range(0, len(ids), batch_size):
                try:
                    batch_ids = ids[i:i+batch_size]
                    batch_docs = documents[i:i+batch_size]
                    batch_metadata = metadatas[i:i+batch_size]
                    batch_embeddings = embeddings[i:i+batch_size]
                    
                    # ChromaDB 0.6.1 - Manejar posibles duplicados
                    try:
                        self.collection.add(
                            ids=batch_ids,
                            documents=batch_docs,
                            metadatas=batch_metadata,
                            embeddings=batch_embeddings
                        )
                        total_added += len(batch_ids)
                        logger.info(f"Lote {i//batch_size + 1} agregado ({len(batch_ids)} chunks)")
                        
                    except Exception as add_error:
                        if "already exists" in str(add_error).lower():
                            logger.warning(f"Algunos chunks ya existen en lote {i//batch_size + 1}, omitiendo...")
                            # Intentar agregar uno por uno para manejar duplicados
                            for j in range(len(batch_ids)):
                                try:
                                    self.collection.add(
                                        ids=[batch_ids[j]],
                                        documents=[batch_docs[j]],
                                        metadatas=[batch_metadata[j]],
                                        embeddings=[batch_embeddings[j]]
                                    )
                                    total_added += 1
                                except:
                                    continue  # Saltar duplicados
                        else:
                            raise add_error
                    
                except Exception as batch_error:
                    logger.error(f"Error en lote {i//batch_size + 1}: {batch_error}")
                    continue
            
            logger.info(f"✅ Documento {document.filename} agregado ({total_added}/{len(ids)} chunks)")
            
        except Exception as e:
            logger.error(f"Error agregando documento {document.filename}: {e}")
            raise
    
    def search_similar(
        self, 
        query: str, 
        n_results: int = 5,
        document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not self.collection:
            logger.error("Colección no inicializada")
            return []
            
        try:
            # Generar embedding de la consulta
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Preparar filtros - ChromaDB 0.6.1 sintaxis
            where_clause = None
            if document_ids:
                where_clause = {"document_id": {"$in": [str(doc_id) for doc_id in document_ids]}}
            
            # Verificar que n_results no exceda el total de documentos
            try:
                collection_count = self.collection.count()
                n_results = min(n_results, collection_count) if collection_count > 0 else n_results
            except:
                pass  # Si no podemos obtener el count, usar n_results original
            
            # Realizar búsqueda
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Formatear resultados
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i] if results['distances'] else 0
                    result = {
                        "id": chunk_id,
                        "content": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "similarity": 1 - distance,  # Convertir distancia a similitud
                        "distance": distance
                    }
                    formatted_results.append(result)
            
            logger.info(f"Búsqueda '{query}' encontró {len(formatted_results)} resultados")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {e}")
            return []
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
            
        try:
            results = self.collection.get(
                where={"document_id": str(document_id)},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            if results['ids']:
                for i, chunk_id in enumerate(results['ids']):
                    chunk = {
                        "id": chunk_id,
                        "content": results['documents'][i],
                        "metadata": results['metadatas'][i]
                    }
                    chunks.append(chunk)
            
            logger.info(f"Obtenidos {len(chunks)} chunks para documento {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error obteniendo chunks del documento {document_id}: {e}")
            return []
    
    def delete_document(self, document_id: str):
        if not self.collection:
            return
            
        try:
            # Obtener IDs de chunks del documento
            results = self.collection.get(
                where={"document_id": str(document_id)},
                include=[]
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"✅ Documento {document_id} eliminado ({len(results['ids'])} chunks)")
            else:
                logger.warning(f"No se encontraron chunks para el documento {document_id}")
                
        except Exception as e:
            logger.error(f"Error eliminando documento {document_id}: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.collection:
            return {"error": "Colección no inicializada"}
            
        try:
            # ChromaDB 0.6.1 - Obtener información básica
            collection_count = self.collection.count()
            
            # Si hay pocos documentos, obtener metadatos completos
            if collection_count < 1000:
                collection_info = self.collection.get(include=["metadatas"])
                
                # Agrupar por documento
                documents = {}
                if collection_info['metadatas']:
                    for metadata in collection_info['metadatas']:
                        doc_id = metadata.get('document_id', 'unknown')
                        if doc_id not in documents:
                            documents[doc_id] = {
                                "filename": metadata.get('filename', 'Unknown'),
                                "chunk_count": 0,
                                "total_words": 0
                            }
                        documents[doc_id]["chunk_count"] += 1
                        documents[doc_id]["total_words"] += metadata.get('word_count', 0)
                
                return {
                    "total_chunks": collection_count,
                    "total_documents": len(documents),
                    "documents": documents,
                    "collection_name": self.collection_name,
                    "client_type": type(self.client).__name__,
                    "chromadb_version": chromadb.__version__
                }
            else:
                return {
                    "total_chunks": collection_count,
                    "total_documents": "N/A (demasiados para contar)",
                    "collection_name": self.collection_name,
                    "client_type": type(self.client).__name__,
                    "chromadb_version": chromadb.__version__
                }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {"error": str(e)}
    
    def clear_collection(self):
        try:
            if self.client:
                self.client.delete_collection(name=self.collection_name)
                self._initialize_collection()
                logger.info("✅ Colección limpiada y reinicializada")
        except Exception as e:
            logger.error(f"Error limpiando colección: {e}")
    
    def hybrid_search(
        self, 
        query: str, 
        n_results: int = 10,
        document_ids: Optional[List[str]] = None,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        try:
            # Búsqueda semántica
            semantic_results = self.search_similar(query, n_results, document_ids)
            
            # Filtrar por threshold de similitud
            filtered_results = [
                result for result in semantic_results 
                if result['similarity'] >= similarity_threshold
            ]
            
            # Si muy pocos resultados, relajar el threshold
            if len(filtered_results) < 3 and semantic_results:
                filtered_results = semantic_results[:max(3, len(semantic_results) // 2)]
            
            logger.info(f"Búsqueda híbrida: {len(filtered_results)} resultados después del filtro")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda híbrida: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        try:
            status = "healthy"
            error_msg = None
            
            # Verificar cliente
            if not self.client:
                status = "unhealthy"
                error_msg = "Cliente no inicializado"
            
            # Verificar colección
            elif not self.collection:
                status = "unhealthy"
                error_msg = "Colección no inicializada"
            
            else:
                # Probar operación básica
                try:
                    count = self.collection.count()
                except Exception as e:
                    status = "degraded"
                    error_msg = f"Error accediendo a colección: {e}"
            
            stats = self.get_collection_stats() if status != "unhealthy" else {}
            
            result = {
                "status": status,
                "client_type": type(self.client).__name__ if self.client else "None",
                "collection_name": self.collection_name,
                "connection": f"{getattr(settings, 'chroma_host', 'unknown')}:{getattr(settings, 'chroma_port', 'unknown')}",
                "chromadb_version": chromadb.__version__,
                "stats": stats
            }
            
            if error_msg:
                result["error"] = error_msg
                
            return result
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "connection": f"{getattr(settings, 'chroma_host', 'unknown')}:{getattr(settings, 'chroma_port', 'unknown')}",
                "chromadb_version": chromadb.__version__
            }