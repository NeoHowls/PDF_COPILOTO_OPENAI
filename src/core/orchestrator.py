
#Orquestador de conversaciones y tareas

from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import re
from datetime import datetime
from models.conversation import ConversationSession, Message, MessageRole, ConversationContext
from models.document import DocumentCollection
from core.vector_store import VectorStore
from services.llm_service import LLMService
from services.analysis_service import AnalysisService


class QueryType(str, Enum):
    GENERAL_QUESTION = "general_question"
    DOCUMENT_SUMMARY = "document_summary"
    DOCUMENT_COMPARISON = "document_comparison"
    SPECIFIC_SEARCH = "specific_search"
    TOPIC_ANALYSIS = "topic_analysis"
    CLARIFICATION = "clarification"


class ConversationOrchestrator:
    
    def __init__(self, vector_store: VectorStore, llm_service: LLMService):
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.analysis_service = AnalysisService(llm_service)
        self.conversation_session: Optional[ConversationSession] = None
        self.document_collection: Optional[DocumentCollection] = None
        
    def initialize_session(self, session: ConversationSession, documents: DocumentCollection):
        #conversacion 
        self.conversation_session = session
        self.document_collection = documents
        
        # Mensaje de bienvenida del sistema
        welcome_message = self._create_welcome_message()
        self.conversation_session.add_message(welcome_message)
    
    def process_user_query(self, query: str) -> Message:

        #consultas
        if not self.conversation_session or not self.document_collection:
            raise ValueError("Sesión no inicializada")
        
        # Crear mensaje del usuario
        user_message = Message(
            id=f"user_{len(self.conversation_session.messages)}",
            role=MessageRole.USER,
            content=query,
            timestamp=datetime.now()
        )
        self.conversation_session.add_message(user_message)
        
        # Clasificar tipo de consulta
        query_type = self._classify_query(query)
        
        # Procesar según el tipo
        response_content, context = self._process_by_type(query, query_type)
        
        # Crear mensaje de respuesta
        assistant_message = Message(
            id=f"assistant_{len(self.conversation_session.messages)}",
            role=MessageRole.ASSISTANT,
            content=response_content,
            timestamp=datetime.now(),
            context=context
        )
        self.conversation_session.add_message(assistant_message)
        
        return assistant_message
    
    def _classify_query(self, query: str) -> QueryType:
        #clasificar consulta
        query_lower = query.lower()
        
        # Patrones para diferentes tipos
        patterns = {
            QueryType.DOCUMENT_SUMMARY: [
                r'\b(resume|resumen|summar|overview|contenido general)\b',
                r'\b(de que trata|que dice|que contiene)\b'
            ],
            QueryType.DOCUMENT_COMPARISON: [
                r'\b(compar|diferenci|similar|contrast|versus|vs)\b',
                r'\b(entre.*y|diferencia entre|similitud entre)\b'
            ],
            QueryType.TOPIC_ANALYSIS: [
                r'\b(tema|topic|categoría|clasificar|agrupar)\b',
                r'\b(sobre que habla|que temas|que categorías)\b'
            ],
            QueryType.SPECIFIC_SEARCH: [
                r'\b(buscar|encontrar|localizar|donde dice|que dice sobre)\b',
                r'\b(información sobre|detalles de|datos de)\b'
            ]
        }
        
        for query_type, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, query_lower):
                    return query_type
        
        return QueryType.GENERAL_QUESTION
    
    def _process_by_type(self, query: str, query_type: QueryType) -> Tuple[str, ConversationContext]:
        #procesar consulta
        context = ConversationContext(last_query_type=query_type.value)
        
        if query_type == QueryType.DOCUMENT_SUMMARY:
            return self._handle_summary_query(query, context)
        elif query_type == QueryType.DOCUMENT_COMPARISON:
            return self._handle_comparison_query(query, context)
        elif query_type == QueryType.TOPIC_ANALYSIS:
            return self._handle_topic_analysis(query, context)
        elif query_type == QueryType.SPECIFIC_SEARCH:
            return self._handle_search_query(query, context)
        else:
            return self._handle_general_query(query, context)
    
    def _handle_summary_query(self, query: str, context: ConversationContext) -> Tuple[str, ConversationContext]:
        # Si se pide resumen de documento específico
        doc_names = self._extract_document_names(query)
        
        if doc_names:
            # Resumen de documentos específicos
            summaries = []
            for doc in self.document_collection.documents:
                if any(name in doc.filename.lower() for name in doc_names):
                    if doc.summary:
                        summaries.append(f"**{doc.filename}:**\n{doc.summary.summary}")
                        context.mentioned_documents.append(doc.id)
            
            if summaries:
                response = "## Resúmenes de documentos:\n\n" + "\n\n".join(summaries)
            else:
                response = "No encontré documentos con esos nombres. Documentos disponibles: " + \
                          ", ".join([doc.filename for doc in self.document_collection.documents])
        else:
            # Resumen general de todos los documentos
            response = self._generate_collection_summary()
            context.mentioned_documents = [doc.id for doc in self.document_collection.documents]
        
        return response, context
    
    def _handle_comparison_query(self, query: str, context: ConversationContext) -> Tuple[str, ConversationContext]:
        if len(self.document_collection.documents) < 2:
            return "Necesitas al menos 2 documentos para hacer comparaciones.", context
        
        # Obtener chunks relevantes de múltiples documentos
        search_results = self.vector_store.hybrid_search(query, n_results=10)
        
        if not search_results:
            return "No encontré información relevante para comparar.", context
        
        # Agrupar por documento
        doc_groups = {}
        for result in search_results:
            doc_id = result['metadata']['document_id']
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(result)
        
        # Generar comparación usando LLM
        comparison_prompt = self._build_comparison_prompt(query, doc_groups)
        response = self.llm_service.generate_response(comparison_prompt)
        
        # Actualizar contexto
        context.mentioned_documents = list(doc_groups.keys())
        context.relevant_chunks = [r['id'] for r in search_results]
        
        return response, context
    
    def _handle_topic_analysis(self, query: str, context: ConversationContext) -> Tuple[str, ConversationContext]:
        # Usar el servicio de análisis
        topic_analysis = self.analysis_service.analyze_topics(self.document_collection)
        
        response_parts = ["## Análisis de Temas:\n"]
        
        for doc in self.document_collection.documents:
            if doc.topics:
                response_parts.append(f"**{doc.filename}:** {', '.join(doc.topics)}")
                context.mentioned_documents.append(doc.id)
        
        if topic_analysis.get('common_themes'):
            response_parts.append(f"\n**Temas comunes:** {', '.join(topic_analysis['common_themes'])}")
        
        response = "\n".join(response_parts)
        context.topics_discussed = topic_analysis.get('all_topics', [])
        
        return response, context
    
    def _handle_search_query(self, query: str, context: ConversationContext) -> Tuple[str, ConversationContext]:
        # Búsqueda en vector store
        search_results = self.vector_store.hybrid_search(query, n_results=5)
        
        if not search_results:
            return "No encontré información relevante para tu consulta.", context
        
        # Construir respuesta con contexto
        response_prompt = self._build_search_response_prompt(query, search_results)
        response = self.llm_service.generate_response(response_prompt)
        
        # Actualizar contexto
        context.relevant_chunks = [r['id'] for r in search_results]
        context.mentioned_documents = list(set([r['metadata']['document_id'] for r in search_results]))
        
        return response, context
    
    def _handle_general_query(self, query: str, context: ConversationContext) -> Tuple[str, ConversationContext]:
        # Buscar información relevante
        search_results = self.vector_store.hybrid_search(query, n_results=7)
        
        if search_results:
            # Respuesta basada en documentos
            response_prompt = self._build_general_response_prompt(query, search_results)
            response = self.llm_service.generate_response(response_prompt)
            
            context.relevant_chunks = [r['id'] for r in search_results]
            context.mentioned_documents = list(set([r['metadata']['document_id'] for r in search_results]))
        else:
            # Respuesta general sin contexto específico
            response = "No encontré información específica en los documentos para responder tu consulta. " + \
                      "¿Podrías ser más específico o reformular la pregunta?"
        
        return response, context
    
    def _create_welcome_message(self) -> Message:
        doc_list = "\n".join([f"• {doc.filename}" for doc in self.document_collection.documents])
        
        content = f"""¡Hola! He procesado {len(self.document_collection.documents)} documento(s) y estoy listo para ayudarte.

**Documentos cargados:**
{doc_list}

**¿Qué puedo hacer por ti?**
• Resumir documentos completos o específicos
• Comparar información entre documentos
• Buscar información específica
• Analizar temas y categorías
• Responder preguntas sobre el contenido

¡Pregúntame lo que necesites saber!"""
        
        return Message(
            id="system_welcome",
            role=MessageRole.SYSTEM,
            content=content,
            timestamp=datetime.now()
        )
    
    def _extract_document_names(self, query: str) -> List[str]:
        """Extraer nombres de documentos mencionados en la consulta"""
        doc_names = []
        query_lower = query.lower()
        
        for doc in self.document_collection.documents:
            # Buscar por nombre completo o parcial
            filename_parts = doc.filename.lower().replace('.pdf', '').split('_')
            for part in filename_parts:
                if len(part) > 3 and part in query_lower:
                    doc_names.append(part)
                    break
        
        return doc_names
    
    def _generate_collection_summary(self) -> str:
        summaries = []
        total_pages = 0
        total_words = 0
        
        for doc in self.document_collection.documents:
            if doc.summary:
                summaries.append(f"**{doc.filename}** ({doc.metadata.page_count} páginas):\n{doc.summary.summary}")
                total_pages += doc.metadata.page_count
                total_words += doc.summary.word_count
        
        header = f"## Resumen General\n**Total:** {len(self.document_collection.documents)} documentos, {total_pages} páginas, ~{total_words:,} palabras\n\n"
        
        return header + "\n\n".join(summaries)
    
    def _build_comparison_prompt(self, query: str, doc_groups: Dict[str, List[Dict]]) -> str:
        prompt_parts = [
            f"Pregunta del usuario: {query}\n",
            "Compara la siguiente información de diferentes documentos:\n"
        ]
        
        for doc_id, chunks in doc_groups.items():
            doc = self.document_collection.get_document_by_id(doc_id)
            doc_name = doc.filename if doc else f"Documento {doc_id}"
            
            prompt_parts.append(f"\n**{doc_name}:**")
            for chunk in chunks[:3]:  # Limitar chunks por documento
                prompt_parts.append(f"- {chunk['content'][:300]}...")
        
        prompt_parts.append("\nProporciona una comparación clara y estructurada.")
        
        return "\n".join(prompt_parts)
    
    def _build_search_response_prompt(self, query: str, search_results: List[Dict]) -> str:
        prompt_parts = [
            f"Pregunta: {query}\n",
            "Información relevante encontrada:\n"
        ]
        
        for i, result in enumerate(search_results, 1):
            doc = self.document_collection.get_document_by_id(result['metadata']['document_id'])
            doc_name = doc.filename if doc else "Documento desconocido"
            
            prompt_parts.append(
                f"\n{i}. **{doc_name}** (Página {result['metadata']['page_number']}):\n"
                f"{result['content']}\n"
            )
        
        prompt_parts.append("Responde la pregunta basándote en la información anterior de manera clara y precisa.")
        
        return "\n".join(prompt_parts)
    
    def _build_general_response_prompt(self, query: str, search_results: List[Dict]) -> str:
        return self._build_search_response_prompt(query, search_results)