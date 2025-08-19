# 🧠 CatchAI - Copiloto Conversacional sobre Documentos

Un sistema avanzado de análisis conversacional de documentos PDF que permite subir hasta 5 archivos y hacer preguntas en lenguaje natural sobre su contenido.

## 🎯 Características Principales

### ✅ Funcionalidades Implementadas
- **📤 Subida de PDFs**: Hasta 5 documentos simultáneos
- **🔍 Extracción y Vectorización**: Procesamiento inteligente del contenido
- **💬 Interfaz Conversacional**: Chat natural con los documentos
- **🎯 Orquestación Estructurada**: Flujo de trabajo extensible y claro
- **📊 Resúmenes Automáticos**: Generación de resúmenes de contenido
- **🔄 Comparaciones entre Documentos**: Análisis cruzado automático
- **🏷️ Clasificación por Temas**: Agrupación inteligente de contenido

### 🌟 Funcionalidades Avanzadas
- **🤖 Múltiples Tipos de Consulta**: Resúmenes, comparaciones, búsquedas específicas
- **📈 Análisis Estadístico**: Insights detallados de la colección
- **🎨 Interfaz Visual Rica**: Gráficos interactivos con Plotly
- **🔧 Configuración Flexible**: Parámetros ajustables
- **⚡ Búsqueda Híbrida**: Combinación de similitud semántica y palabras clave

## 🏗️ Arquitectura del Sistema

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Streamlit UI   │────│  Orchestrator    │────│   LLM Service   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Document        │    │   Vector Store   │    │ Analysis        │
│ Processor       │    │   (ChromaDB)     │    │ Service         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### 🧩 Componentes Principales

#### 1. **Core Module** - Motor Principal
- **`DocumentProcessor`**: Extracción y división de PDFs
- **`VectorStore`**: Manejo de embeddings con ChromaDB
- **`ConversationOrchestrator`**: Orquestación inteligente de conversaciones

#### 2. **Services Module** - Servicios Especializados
- **`LLMService`**: Abstracción para OpenAI/Anthropic/Mock
- **`AnalysisService`**: Análisis avanzado y comparaciones
- **`EmbeddingService`**: Generación de embeddings

#### 3. **Models Module** - Modelos de Datos
- **`Document`**: Estructura de documentos y chunks
- **`Conversation`**: Gestión de sesiones y mensajes

## 🛠️ Stack Tecnológico

### **Orquestación y LLM**
- **LangChain**: Framework de orquestación
- **OpenAI GPT-3.5/4**: Modelo de lenguaje principal
- **Anthropic Claude**: Modelo alternativo
- **Sentence Transformers**: Embeddings locales

### **Vector Store**
- **ChromaDB**: Base de datos vectorial
- **FAISS**: Búsqueda de similitud (fallback)

### **Interfaz y Backend**
- **Streamlit**: Framework web interactivo
- **Plotly**: Visualizaciones interactivas
- **Pandas**: Análisis de datos

### **Procesamiento**
- **PyPDF2**: Extracción de texto de PDFs
- **scikit-learn**: Clustering y análisis
- **langdetect**: Detección de idioma

### **Contenerización**
- **Docker**: Contenerización de la aplicación
- **docker-compose**: Orquestación de servicios

## 🚀 Instalación y Uso

### **Opción 1: Docker (Recomendado)**

```bash
# 1. Clonar repositorio
git clone <repository-url>
cd pdf-copilot

# 2. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus API keys

# 3. Levantar servicios
docker-compose up --build

# 4. Abrir aplicación
# http://localhost:8501
```

### **Opción 2: Instalación Local**

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar variables
cp .env.example .env

# 4. Iniciar ChromaDB (separado)
chroma run --path ./data/vectorstore

# 5. Ejecutar aplicación
streamlit run src/app.py
```

## ⚙️ Configuración

### **Variables de Entorno Clave**

```bash
# APIs de LLM (al menos una)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Límites de archivos
MAX_PDFS=5
MAX_FILE_SIZE_MB=10

# Procesamiento de texto
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Vector Store
CHROMA_HOST=chromadb
CHROMA_PORT=8000
```

## 💡 Flujo Conversacional

### **1. Clasificación de Consultas**
```python
QueryType = {
    "GENERAL_QUESTION",    # Pregunta general
    "DOCUMENT_SUMMARY",    # Solicitud de resumen
    "DOCUMENT_COMPARISON", # Comparación entre docs
    "SPECIFIC_SEARCH",     # Búsqueda específica
    "TOPIC_ANALYSIS"       # Análisis temático
}
```

### **2. Orquestación de Respuestas**
```
Usuario → Clasificador → Procesador Específico → LLM → Respuesta
                ↓
        Vector Store ← Búsqueda Semántica
```

### **3. Contexto Conversacional**
- **Historial de mensajes**
- **Documentos referenciados**
- **Chunks relevantes utilizados**
- **Temas discutidos**

## 📊 Funcionalidades de Análisis

### **Resúmenes Automáticos**
- Resumen por documento individual
- Resumen de colección completa
- Puntos clave extraídos automáticamente

### **Comparación de Documentos**
- Similitud de contenido (TF-IDF + Cosine)
- Solapamiento de temas (Jaccard)
- Análisis de diferencias usando LLM

### **Clasificación Temática**
- Clustering automático (K-means)
- Extracción de palabras clave
- Análisis de frecuencia de temas

### **Métricas y Estadísticas**
- Distribución de tamaños
- Análisis de similitudes
- Insights de colección

## 🧪 Ejemplos de Uso

### **Consultas Soportadas**

```
📋 Resúmenes
"Resume el documento X"
"¿De qué trata la colección?"
"Dame un overview de todos los documentos"

🔍 Búsquedas Específicas  
"¿Qué dice sobre inteligencia artificial?"
"Busca información sobre metodología"
"Encuentra referencias a casos de estudio"

⚖️ Comparaciones
"Compara los documentos X e Y"
"¿Cuáles son las diferencias entre estos docs?"
"¿Qué tienen en común todos los documentos?"

🏷️ Análisis Temático
"¿Qué temas principales cubren?"
"Clasifica los documentos por categorías"
"¿Cuáles son los conceptos más frecuentes?"
```

## 🔧 Desarrollo y Extensibilidad

### **Agregar Nuevos Tipos de Consulta**

```python
# 1. Añadir al enum QueryType
class QueryType(str, Enum):
    NEW_QUERY_TYPE = "new_query_type"

# 2. Implementar en Orchestrator
def _handle_new_query_type(self, query: str, context: ConversationContext):
    # Lógica específica
    return response, context

# 3. Agregar al clasificador
patterns = {
    QueryType.NEW_QUERY_TYPE: [r'\b(patron1|patron2)\b']
}
```

### **Integrar Nuevos LLMs**

```python
class NewLLMService(BaseLLMService):
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> str:
        # Implementación específica
        pass
    
    def generate_summary(self, text: str, max_length: int = 200) -> str:
        # Implementación específica
        pass
```

### **Extender Análisis**

```python
class CustomAnalysisService(AnalysisService):
    def custom_analysis_method(self, documents):
        # Nueva funcionalidad de análisis
        pass
```

## 🧪 Testing

### **Ejecutar Tests**

```bash
# Todos los tests
pytest tests/

# Tests específicos
pytest tests/test_document_processor.py
pytest tests/test_vector_store.py
pytest tests/test_orchestrator.py

# Con cobertura
pytest --cov=src tests/
```

### **Tests Implementados**
- ✅ Procesamiento de PDFs
- ✅ Vector Store operations
- ✅ Orquestación de conversaciones
- ✅ Servicios de LLM
- ✅ Análisis de documentos

## 📈 Rendimiento y Escalabilidad

### **Métricas de Performance**
- **Tiempo de procesamiento**: ~2-5s por PDF (dependiendo del tamaño)
- **Memoria RAM**: ~500MB base + ~100MB por documento
- **Vector Store**: ChromaDB optimizado para consultas rápidas
- **Concurrencia**: Soporte para múltiples usuarios (con limitaciones)

### **Optimizaciones Implementadas**
- ✅ Caching de embeddings
- ✅ Chunking inteligente con solapamiento
- ✅ Búsqueda híbrida optimizada
- ✅ Lazy loading de modelos
- ✅ Streaming de respuestas LLM

## 🚨 Limitaciones Actuales

### **Técnicas**
- **Límite de archivos**: 5 PDFs máximo
- **Tamaño máximo**: 10MB por archivo
- **Tipos de archivo**: Solo PDFs con texto extraíble
- **Idiomas**: Optimizado para español e inglés
- **Concurrencia**: Una sesión por instancia

### **Funcionales**
- **PDFs con imágenes**: Texto en imágenes no se extrae
- **Tablas complejas**: Formato puede perderse
- **Documentos encriptados**: No soportados
- **Formatos mixtos**: Solo PDFs

## 🛣️ Roadmap y Mejoras Futuras

### **Corto Plazo (v2.0)**
- [ ] **OCR Integration**: Extracción de texto de imágenes
- [ ] **Multi-idioma**: Soporte extendido para más idiomas
- [ ] **Autenticación**: Sistema de usuarios
- [ ] **Persistencia**: Sesiones guardadas
- [ ] **API REST**: Endpoints para integración

### **Medio Plazo (v3.0)**
- [ ] **Múltiples Formatos**: Word, Excel, PowerPoint
- [ ] **Análisis de Imágenes**: Descripción automática
- [ ] **Grafos de Conocimiento**: Relaciones entre conceptos
- [ ] **Exportación**: Reportes en PDF/Word
- [ ] **Colaboración**: Sesiones compartidas

### **Largo Plazo (v4.0)**
- [ ] **IA Multimodal**: Análisis de audio y video
- [ ] **Agentes Autónomos**: Tareas complejas automatizadas
- [ ] **Integración Cloud**: AWS/GCP/Azure
- [ ] **Escalabilidad**: Arquitectura distribuida
- [ ] **Personalización**: Modelos fine-tuned por dominio

## 🔒 Consideraciones de Seguridad

### **Datos y Privacidad**
- **Archivos locales**: Los PDFs se procesan localmente
- **Vector Store**: Datos almacenados en contenedor local
- **API Keys**: Manejo seguro mediante variables de entorno
- **Sin persistencia cloud**: Los datos no salen del entorno local

### **Recomendaciones**
- ✅ Usar `.env` para API keys
- ✅ No incluir documentos sensibles en repos
- ✅ Limpiar datos regularmente
- ✅ Usar HTTPS en producción

## 📚 Documentación Adicional

### **Estructura de Archivos**
```
src/
├── config/          # Configuración centralizada
├── core/            # Lógica principal del negocio
├── models/          # Modelos de datos (Pydantic)
├── services/        # Servicios especializados
└── utils/           # Utilidades y helpers
```

### **Modelos de Datos Clave**
- **`Document`**: Representación completa de un PDF procesado
- **`TextChunk`**: Fragmento de texto con embeddings
- **`ConversationSession`**: Sesión de chat con contexto
- **`Message`**: Mensaje individual con metadatos

### **Patrones de Diseño Utilizados**
- **Factory Pattern**: `LLMService` para diferentes proveedores
- **Strategy Pattern**: Diferentes tipos de consulta
- **Observer Pattern**: Estado de sesión reactivo
- **Repository Pattern**: Acceso a datos del vector store

## 🤝 Contribución

### **Cómo Contribuir**
1. **Fork** el repositorio
2. **Crear branch** para tu feature: `git checkout -b feature/nueva-funcionalidad`
3. **Commit** cambios: `git commit -m 'Añadir nueva funcionalidad'`
4. **Push** al branch: `git push origin feature/nueva-funcionalidad`
5. **Crear Pull Request**

### **Estándares de Código**
- **Linting**: Black + flake8
- **Type Hints**: Obligatorios en funciones públicas
- **Docstrings**: Documentación clara y concisa
- **Tests**: Cobertura mínima del 80%

### **Áreas que Necesitan Contribución**
- 🔧 Optimización de rendimiento
- 🌐 Internacionalización
- 🧪 Más casos de test
- 📖 Documentación y ejemplos
- 🎨 Mejoras de UX/UI

## 📞 Soporte y Contacto

### **Issues y Bugs**
- Reportar en GitHub Issues
- Incluir logs y pasos para reproducir
- Especificar versión y entorno

### **Preguntas y Discusión**
- GitHub Discussions para preguntas generales
- Stack Overflow con tag `catchai-copilot`

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **LangChain**: Framework de orquestación
- **ChromaDB**: Vector database eficiente  
- **Streamlit**: Interfaz web rápida y elegante
- **OpenAI/Anthropic**: APIs de modelos de lenguaje
- **Sentence Transformers**: Embeddings de alta calidad

---

**🧠 CatchAI Copiloto** - Transforma tus documentos en conocimiento conversacional

*Desarrollado con ❤️ para el desafío técnico CatchAI*