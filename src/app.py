
import streamlit as st
import os
import uuid
from datetime import datetime
from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Imports del proyecto
from core.document_processor import DocumentProcessor
from core.vector_store import VectorStore
from core.orchestrator import ConversationOrchestrator
from models.document import DocumentCollection
from models.conversation import ConversationSession, Message, MessageRole
from services.llm_service import LLMService
from config.settings import settings


# Configuración de la página
st.set_page_config(
    page_title="🧠 CatchAI - Copiloto PDF",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.stAlert > div {
    padding: 0.5rem;
}
.document-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background: #f8f9fa;
}
.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 8px;
    border-left: 4px solid #007acc;
    background: #f0f8ff;
}
.user-message {
    border-left-color: #28a745;
    background: #f8fff8;
}
.assistant-message {
    border-left-color: #007acc;
    background: #f0f8ff;
}
.stats-box {
    background: #e9ecef;
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_services():
    vector_store = VectorStore()
    llm_service = LLMService()
    document_processor = DocumentProcessor()
    
    return vector_store, llm_service, document_processor


def initialize_session_state():
    """Inicializar estado de la sesión"""
    if "conversation_session" not in st.session_state:
        st.session_state.conversation_session = None
        
    if "document_collection" not in st.session_state:
        st.session_state.document_collection = DocumentCollection()
        
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
        
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    
    # Agregar estado para el tab activo
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "chat" if st.session_state.document_collection.documents else "upload"


def render_sidebar():
    with st.sidebar:
        st.header("🧠 CatchAI Copiloto")
        st.markdown("---")
        
        # Información de la sesión
        if st.session_state.conversation_session:
            st.success("✅ Sesión activa")
            st.info(f"📄 {len(st.session_state.document_collection.documents)} documentos cargados")
            st.info(f"💬 {len(st.session_state.conversation_session.messages)} mensajes")
        else:
            st.warning("⏳ Sin sesión activa")
        
        st.markdown("---")
        
        # Navegación manual
        st.subheader("🧭 Navegación")
        
        nav_col1, nav_col2, nav_col3 = st.columns(3)
        with nav_col1:
            if st.button("💬", help="Chat", use_container_width=True):
                st.session_state.active_tab = "chat"
                st.rerun()
        with nav_col2:
            if st.button("📚", help="Documentos", use_container_width=True):
                st.session_state.active_tab = "docs"
                st.rerun()
        with nav_col3:
            if st.button("📤", help="Cargar", use_container_width=True):
                st.session_state.active_tab = "upload"
                st.rerun()
        
        st.markdown("---")
        
        # Estadísticas de documentos
        if st.session_state.document_collection.documents:
            st.subheader("📊 Estadísticas")
            
            total_pages = sum(doc.metadata.page_count for doc in st.session_state.document_collection.documents)
            total_chunks = len(st.session_state.document_collection.get_all_chunks())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Páginas", total_pages)
            with col2:
                st.metric("Chunks", total_chunks)
        
        st.markdown("---")
        
        # Controles de sesión
        st.subheader("🔧 Controles")
        
        if st.button("🔄 Nueva Sesión", help="Iniciar nueva sesión limpia"):
            reset_session()
            st.rerun()
        
        if st.button("🗑️ Limpiar Vector Store", help="Limpiar base de datos vectorial"):
            if st.session_state.get("vector_store"):
                st.session_state.vector_store.clear_collection()
                st.success("Vector store limpiado")
                st.rerun()
        
        # Configuración avanzada
        with st.expander("⚙️ Configuración Avanzada"):
            st.slider("Max PDFs", min_value=1, max_value=10, value=settings.max_pdfs, disabled=True)
            st.slider("Chunk Size", min_value=500, max_value=2000, value=settings.chunk_size, disabled=True)
            st.slider("Chunk Overlap", min_value=50, max_value=500, value=settings.chunk_overlap, disabled=True)


def render_document_upload():
    st.header("📄 Cargar Documentos PDF")
    
    # Verificar límite de documentos
    current_doc_count = len(st.session_state.document_collection.documents)
    remaining_slots = settings.max_pdfs - current_doc_count
    
    if remaining_slots <= 0:
        st.warning(f"⚠️ Has alcanzado el límite máximo de {settings.max_pdfs} documentos.")
        return
    
    # Cargador de archivos
    uploaded_files = st.file_uploader(
        f"Selecciona hasta {remaining_slots} archivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Máximo {settings.max_file_size_mb}MB por archivo"
    )
    
    if uploaded_files:
        if len(uploaded_files) > remaining_slots:
            st.error(f"❌ Solo puedes cargar {remaining_slots} archivo(s) más.")
            return
        
        if st.button("📤 Procesar Documentos", type="primary"):
            process_uploaded_files(uploaded_files)


def process_uploaded_files(uploaded_files):
    vector_store, llm_service, document_processor = initialize_services()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processed_count = 0
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Actualizar progreso
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            status_text.text(f"Procesando {uploaded_file.name}...")
            
            # Guardar archivo temporalmente
            temp_path = f"data/uploads/{uuid.uuid4()}_{uploaded_file.name}"
            os.makedirs("data/uploads", exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Validar archivo
            validation = document_processor.validate_pdf(temp_path)
            if not validation["is_valid"]:
                st.error(f"❌ {uploaded_file.name}: {', '.join(validation['errors'])}")
                os.remove(temp_path)
                continue
            
            # Procesar documento
            document = document_processor.process_pdf(temp_path, uploaded_file.name)
            
            # Agregar a la colección
            st.session_state.document_collection.add_document(document)
            
            # Agregar al vector store
            vector_store.add_document(document)
            
            processed_count += 1
            st.success(f"✅ {uploaded_file.name} procesado correctamente")
            
            # Limpiar archivo temporal
            os.remove(temp_path)
            
        except Exception as e:
            st.error(f"❌ Error procesando {uploaded_file.name}: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    progress_bar.empty()
    status_text.empty()
    
    if processed_count > 0:
        st.success(f"🎉 {processed_count} documento(s) procesado(s) exitosamente!")
        
        # Inicializar sesión de conversación
        initialize_conversation_session()
        # Cambiar a tab de chat automáticamente
        st.session_state.active_tab = "chat"
        st.rerun()


def initialize_conversation_session():
    vector_store, llm_service, document_processor = initialize_services()
    
    session = ConversationSession(
        id=str(uuid.uuid4()),
        document_ids=[doc.id for doc in st.session_state.document_collection.documents],
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    
    orchestrator = ConversationOrchestrator(vector_store, llm_service)
    orchestrator.initialize_session(session, st.session_state.document_collection)
    
    st.session_state.conversation_session = session
    st.session_state.orchestrator = orchestrator
    st.session_state.vector_store = vector_store


def render_document_overview():
    if not st.session_state.document_collection.documents:
        st.info("📝 No hay documentos cargados aún.")
        return
    
    st.header("📚 Documentos Cargados")
    
    # Sub-tabs para diferentes vistas (ahora usando radio buttons)
    view_option = st.radio("Seleccionar vista:", ["📋 Lista", "📊 Estadísticas", "🔍 Análisis"], horizontal=True)
    
    if view_option == "📋 Lista":
        render_document_list()
    elif view_option == "📊 Estadísticas":
        render_document_statistics()
    elif view_option == "🔍 Análisis":
        render_document_analysis()


def render_document_list():
    for i, doc in enumerate(st.session_state.document_collection.documents):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**📄 {doc.filename}**")
                if doc.summary:
                    st.markdown(f"*{doc.summary.summary[:200]}...*")
                
                if doc.topics:
                    st.markdown("**Temas:** " + ", ".join(f"`{topic}`" for topic in doc.topics[:5]))
            
            with col2:
                st.metric("Páginas", doc.metadata.page_count)
                if doc.summary:
                    st.metric("Palabras", f"{doc.summary.word_count:,}")
            
            with col3:
                st.metric("Chunks", len(doc.chunks))
                processing_time = doc.metadata.processing_time or 0
                st.metric("Tiempo proc.", f"{processing_time:.1f}s")
        
        st.markdown("---")


def render_document_statistics():
    if not st.session_state.document_collection.documents:
        return
    
    # Preparar datos
    docs_data = []
    for doc in st.session_state.document_collection.documents:
        docs_data.append({
            "Documento": doc.filename,
            "Páginas": doc.metadata.page_count,
            "Palabras": doc.summary.word_count if doc.summary else 0,
            "Chunks": len(doc.chunks),
            "Temas": len(doc.topics)
        })
    
    df = pd.DataFrame(docs_data)
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pages = px.bar(df, x="Documento", y="Páginas", 
                          title="Páginas por Documento",
                          color="Páginas", color_continuous_scale="viridis")
        fig_pages.update_xaxis(tickangle=45)
        st.plotly_chart(fig_pages, use_container_width=True)
    
    with col2:
        fig_words = px.bar(df, x="Documento", y="Palabras", 
                          title="Palabras por Documento",
                          color="Palabras", color_continuous_scale="plasma")
        fig_words.update_xaxis(tickangle=45)
        st.plotly_chart(fig_words, use_container_width=True)
    
    # Tabla de estadísticas
    st.subheader("📋 Resumen Estadístico")
    st.dataframe(df, use_container_width=True)


def render_document_analysis():
    if not st.session_state.document_collection.documents:
        return
    
    if not st.session_state.orchestrator:
        st.warning("⚠️ Inicializa la conversación primero.")
        return
    
    with st.spinner("🔍 Analizando documentos..."):
        analysis_service = st.session_state.orchestrator.analysis_service
        insights = analysis_service.generate_collection_insights(st.session_state.document_collection)
    
    if "error" in insights:
        st.error(f"❌ {insights['error']}")
        return
    
    # Mostrar insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Estadísticas Básicas")
        stats = insights["basic_stats"]
        
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Total Documentos", stats["total_documents"])
            st.metric("Total Palabras", f"{stats['total_words']:,}")
        with metrics_col2:
            st.metric("Total Páginas", stats["total_pages"])
            st.metric("Promedio Páginas", stats["avg_pages_per_doc"])
    
    with col2:
        st.subheader("📈 Distribución de Tamaños")
        if "size_distribution" in insights:
            size_dist = insights["size_distribution"]
            st.metric("Páginas Mínimas", size_dist["min_pages"])
            st.metric("Páginas Máximas", size_dist["max_pages"])
            st.metric("Mediana", size_dist["median_pages"])
    
    # Análisis de temas
    if "topic_analysis" in insights and not insights["topic_analysis"].get("error"):
        st.subheader("🏷️ Análisis de Temas")
        topic_data = insights["topic_analysis"]
        
        if "common_themes" in topic_data:
            st.markdown("**Temas Más Comunes:**")
            for i, theme in enumerate(topic_data["common_themes"][:10], 1):
                st.markdown(f"{i}. `{theme}`")


def render_chat_interface():
    if not st.session_state.conversation_session:
        st.info("💬 Carga algunos documentos para comenzar la conversación.")
        return
    
    st.header("💬 Conversación")
    
    # Contenedor de mensajes
    chat_container = st.container()
    
    # Renderizar historial de mensajes
    with chat_container:
        for message in st.session_state.conversation_session.messages:
            render_message(message)


def render_message(message: Message):
    if message.role == MessageRole.USER:
        with st.chat_message("user"):
            st.markdown(message.content)
    
    elif message.role == MessageRole.ASSISTANT:
        with st.chat_message("assistant"):
            st.markdown(message.content)
            
            # Mostrar contexto si existe
            if message.context and message.context.relevant_chunks:
                with st.expander("🔍 Fuentes consultadas"):
                    st.info(f"📄 {len(message.context.mentioned_documents)} documento(s) consultado(s)")
                    st.info(f"📝 {len(message.context.relevant_chunks)} fragmento(s) relevante(s)")
    
    elif message.role == MessageRole.SYSTEM:
        with st.chat_message("assistant"):
            st.markdown(message.content)


def handle_user_message(user_input: str):
    if not st.session_state.orchestrator:
        st.error("❌ Error: Orquestador no inicializado")
        return
    
    with st.spinner("🤔 Procesando tu consulta..."):
        try:
            # Procesar consulta
            response_message = st.session_state.orchestrator.process_user_query(user_input)
            
            # Forzar actualización de la interfaz
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error procesando consulta: {str(e)}")


def reset_session():
    # Limpiar vector store si existe
    if "vector_store" in st.session_state:
        st.session_state.vector_store.clear_collection()
    
    # Resetear estado
    st.session_state.conversation_session = None
    st.session_state.document_collection = DocumentCollection()
    st.session_state.orchestrator = None
    st.session_state.uploaded_files = []
    st.session_state.active_tab = "upload"
    
    # Limpiar archivos temporales
    uploads_dir = "data/uploads"
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            file_path = os.path.join(uploads_dir, filename)
            try:
                os.remove(file_path)
            except:
                pass


def main():
    initialize_session_state()
    
    # Título principal
    st.title("🧠 CatchAI - Copiloto Conversacional sobre Documentos")
    st.markdown("*Sube documentos PDF y haz preguntas en lenguaje natural*")
    st.markdown("---")
    
    # Renderizar sidebar
    render_sidebar()
    
    # Contenido principal - Mostrar según tab activo
    if st.session_state.active_tab == "chat":
        render_chat_interface()
    elif st.session_state.active_tab == "docs":
        render_document_overview()
    elif st.session_state.active_tab == "upload":
        render_document_upload()
    
    # Input de chat FUERA de cualquier contenedor
    # Solo mostrar si hay una sesión de conversación activa y estamos en el tab de chat
    if st.session_state.conversation_session and st.session_state.active_tab == "chat":
        user_input = st.chat_input("Escribe tu pregunta aquí...")
        
        if user_input:
            handle_user_message(user_input)


if __name__ == "__main__":
    # Verificar configuración
    if not settings.openai_api_key and not settings.anthropic_api_key:
        st.error("❌ **Error de Configuración**: No se encontraron API keys.")
        st.info("📝 Crea un archivo `.env` con tu API key de OpenAI o Anthropic.")
        st.code("""
# .env
OPENAI_API_KEY=tu_clave_aqui
# o
ANTHROPIC_API_KEY=tu_clave_aqui
        """)
        st.stop()
    
    main()