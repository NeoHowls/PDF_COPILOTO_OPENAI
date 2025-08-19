
from typing import List, Dict, Any, Set, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from models.document import DocumentCollection, Document
from services.llm_service import LLMService


class AnalysisService:
    
    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def analyze_topics(self, document_collection: DocumentCollection) -> Dict[str, Any]:
        if not document_collection.documents:
            return {"error": "No hay documentos para analizar"}
        
        # Recopilar todos los temas de los documentos
        all_topics = []
        document_topics = {}
        
        for doc in document_collection.documents:
            if doc.topics:
                all_topics.extend(doc.topics)
                document_topics[doc.id] = doc.topics
        
        if not all_topics:
            return {"error": "No se encontraron temas en los documentos"}
        
        # Análisis de frecuencia de temas
        topic_freq = Counter(all_topics)
        common_themes = [topic for topic, count in topic_freq.most_common(10)]
        
        # Análisis de similitud entre documentos basado en temas
        similarity_matrix = self._calculate_topic_similarity(document_topics)
        
        return {
            "all_topics": list(set(all_topics)),
            "topic_frequency": dict(topic_freq),
            "common_themes": common_themes,
            "document_topics": document_topics,
            "similarity_matrix": similarity_matrix,
            "total_unique_topics": len(set(all_topics))
        }
    
    def compare_documents(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        comparison = {
            "basic_stats": self._compare_basic_stats(doc1, doc2),
            "topic_overlap": self._calculate_topic_overlap(doc1, doc2),
            "content_similarity": self._calculate_content_similarity(doc1, doc2),
            "summary_comparison": self._compare_summaries(doc1, doc2)
        }
        
        return comparison
    
    def classify_document_themes(self, document_collection: DocumentCollection, n_clusters: int = 3) -> Dict[str, Any]:
        if len(document_collection.documents) < 2:
            return {"error": "Se necesitan al menos 2 documentos para clasificar"}
        
        # Preparar textos para vectorización
        documents_text = []
        doc_ids = []
        
        for doc in document_collection.documents:
            # Combinar resumen y temas como texto representativo
            text_parts = []
            if doc.summary:
                text_parts.append(doc.summary.summary)
            if doc.topics:
                text_parts.append(" ".join(doc.topics))
            
            if text_parts:
                documents_text.append(" ".join(text_parts))
                doc_ids.append(doc.id)
        
        if len(documents_text) < n_clusters:
            n_clusters = len(documents_text)
        
        # Vectorización TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(documents_text)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Organizar resultados
        clustered_docs = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_docs:
                clustered_docs[cluster_id] = []
            
            doc = document_collection.get_document_by_id(doc_ids[i])
            if doc:
                clustered_docs[cluster_id].append({
                    "id": doc.id,
                    "filename": doc.filename,
                    "topics": doc.topics
                })
        
        # Generar nombres para los clusters
        cluster_names = self._generate_cluster_names(clustered_docs)
        
        return {
            "clusters": clustered_docs,
            "cluster_names": cluster_names,
            "n_clusters": n_clusters,
            "silhouette_score": self._calculate_silhouette_score(tfidf_matrix, clusters)
        }
    
    def generate_collection_insights(self, document_collection: DocumentCollection) -> Dict[str, Any]:
        if not document_collection.documents:
            return {"error": "No hay documentos para analizar"}
        
        insights = {}
        
        # Estadísticas básicas
        total_pages = sum(doc.metadata.page_count for doc in document_collection.documents)
        total_words = sum(doc.summary.word_count if doc.summary else 0 for doc in document_collection.documents)
        avg_pages = total_pages / len(document_collection.documents)
        
        insights["basic_stats"] = {
            "total_documents": len(document_collection.documents),
            "total_pages": total_pages,
            "total_words": total_words,
            "avg_pages_per_doc": round(avg_pages, 1),
            "avg_words_per_doc": round(total_words / len(document_collection.documents), 0) if document_collection.documents else 0
        }
        
        # Análisis de temas
        topic_analysis = self.analyze_topics(document_collection)
        insights["topic_analysis"] = topic_analysis
        
        # Documentos más similares y más diferentes
        if len(document_collection.documents) >= 2:
            similarities = self._find_document_similarities(document_collection)
            insights["document_relationships"] = similarities
        
        # Distribución de tamaños
        page_counts = [doc.metadata.page_count for doc in document_collection.documents]
        insights["size_distribution"] = {
            "min_pages": min(page_counts),
            "max_pages": max(page_counts),
            "median_pages": np.median(page_counts),
            "std_pages": round(np.std(page_counts), 1)
        }
        
        return insights
    
    def _compare_basic_stats(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        return {
            "page_count_diff": abs(doc1.metadata.page_count - doc2.metadata.page_count),
            "word_count_diff": abs(
                (doc1.summary.word_count if doc1.summary else 0) - 
                (doc2.summary.word_count if doc2.summary else 0)
            ),
            "size_ratio": doc1.metadata.page_count / max(doc2.metadata.page_count, 1)
        }
    
    def _calculate_topic_overlap(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        topics1 = set(doc1.topics) if doc1.topics else set()
        topics2 = set(doc2.topics) if doc2.topics else set()
        
        overlap = topics1.intersection(topics2)
        union = topics1.union(topics2)
        
        jaccard = len(overlap) / len(union) if union else 0
        
        return {
            "common_topics": list(overlap),
            "unique_to_doc1": list(topics1 - topics2),
            "unique_to_doc2": list(topics2 - topics1),
            "jaccard_similarity": round(jaccard, 3),
            "overlap_count": len(overlap)
        }
    
    def _calculate_content_similarity(self, doc1: Document, doc2: Document) -> float:
        if not doc1.summary or not doc2.summary:
            return 0.0
        
        # Usar TF-IDF para calcular similitud
        texts = [doc1.summary.summary, doc2.summary.summary]
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return round(similarity, 3)
    
    def _compare_summaries(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        if not doc1.summary or not doc2.summary:
            return {"error": "Uno o ambos documentos no tienen resumen"}
        
        comparison_prompt = f"""
        Compara estos dos resúmenes de documentos y proporciona un análisis breve:

        Documento 1 ({doc1.filename}):
        {doc1.summary.summary}

        Documento 2 ({doc2.filename}):
        {doc2.summary.summary}

        Identifica:
        1. Similitudes principales
        2. Diferencias clave
        3. Temas en común
        4. Enfoques únicos de cada documento
        """
        
        comparison = self.llm_service.generate_response(comparison_prompt, max_tokens=500)
        
        return {
            "llm_comparison": comparison,
            "doc1_length": len(doc1.summary.summary),
            "doc2_length": len(doc2.summary.summary)
        }
    
    def _calculate_topic_similarity(self, document_topics: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        doc_ids = list(document_topics.keys())
        similarity_matrix = {}
        
        for i, doc_id1 in enumerate(doc_ids):
            similarity_matrix[doc_id1] = {}
            for j, doc_id2 in enumerate(doc_ids):
                if i == j:
                    similarity_matrix[doc_id1][doc_id2] = 1.0
                else:
                    topics1 = set(document_topics[doc_id1])
                    topics2 = set(document_topics[doc_id2])
                    
                    intersection = len(topics1.intersection(topics2))
                    union = len(topics1.union(topics2))
                    
                    jaccard = intersection / union if union > 0 else 0
                    similarity_matrix[doc_id1][doc_id2] = round(jaccard, 3)
        
        return similarity_matrix
    
    def _generate_cluster_names(self, clustered_docs: Dict[int, List[Dict]]) -> Dict[int, str]:
        cluster_names = {}
        
        for cluster_id, docs in clustered_docs.items():
            # Recopilar todos los temas del cluster
            cluster_topics = []
            for doc in docs:
                if doc.get("topics"):
                    cluster_topics.extend(doc["topics"])
            
            # Encontrar temas más comunes
            if cluster_topics:
                topic_freq = Counter(cluster_topics)
                top_topics = [topic for topic, count in topic_freq.most_common(3)]
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}: {', '.join(top_topics)}"
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id + 1}: Sin temas definidos"
        
        return cluster_names
    
    def _calculate_silhouette_score(self, tfidf_matrix, clusters) -> float:
        try:
            from sklearn.metrics import silhouette_score
            if len(set(clusters)) > 1:
                score = silhouette_score(tfidf_matrix.toarray(), clusters)
                return round(score, 3)
        except:
            pass
        return 0.0
    
    def _find_document_similarities(self, document_collection: DocumentCollection) -> Dict[str, Any]:
        if len(document_collection.documents) < 2:
            return {}
        
        similarities = []
        
        docs = document_collection.documents
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                doc1, doc2 = docs[i], docs[j]
                
                # Calcular similitud basada en temas
                topic_overlap = self._calculate_topic_overlap(doc1, doc2)
                content_similarity = self._calculate_content_similarity(doc1, doc2)
                
                combined_similarity = (topic_overlap["jaccard_similarity"] + content_similarity) / 2
                
                similarities.append({
                    "doc1": {"id": doc1.id, "filename": doc1.filename},
                    "doc2": {"id": doc2.id, "filename": doc2.filename},
                    "similarity": combined_similarity,
                    "topic_overlap": topic_overlap["overlap_count"],
                    "content_similarity": content_similarity
                })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "most_similar": similarities[0] if similarities else None,
            "least_similar": similarities[-1] if similarities else None,
            "all_similarities": similarities
        }