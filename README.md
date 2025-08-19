# CatchAI - Copiloto Conversacional sobre Documentos

Un sistema avanzado de análisis conversacional de documentos PDF que permite subir hasta 5 archivos y hacer preguntas en lenguaje natural sobre su contenido.

## Características Principales

### Funcionalidades Implementadas
- **Subida de PDFs**: Hasta 5 documentos simultáneos
- **Extracción y Vectorización**: Procesamiento inteligente del contenido
- **Interfaz Conversacional**: Chat natural con los documentos
- **Orquestación Estructurada**: Flujo de trabajo extensible y claro
- **Resúmenes Automáticos**: Generación de resúmenes de contenido
- **Comparaciones entre Documentos**: Análisis cruzado automático
- **Clasificación por Temas**: Agrupación inteligente de contenido

### Funcionalidades Avanzadas
- **Múltiples Tipos de Consulta**: Resúmenes, comparaciones, búsquedas específicas
- **Análisis Estadístico**: Insights detallados de la colección
- **Interfaz Visual Rica**: Gráficos interactivos con Plotly
- **Configuración Flexible**: Parámetros ajustables
- **Búsqueda Híbrida**: Combinación de similitud semántica y palabras clave

## Arquitectura del Sistema

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

### Componentes Principales

#### 1. Core Module - Motor Principal
- **`DocumentProcessor`**: Extracción y división de PDFs
- **`VectorStore`**: Manejo de embeddings con ChromaDB
- **`ConversationOrchestrator`**: Orquestación inteligente de conversaciones

#### 2. Services Module - Servicios Especializados
- **`LLMService`**: Abstracción para OpenAI/Anthropic/Mock
- **`AnalysisService`**: Análisis avanzado y comparaciones
- **`EmbeddingService`**: Generación de embeddings

#### 3. Models Module - Modelos de Datos
- **`Document`**: Estructura de documentos y chunks
- **`Conversation`**: Gestión de sesiones y mensajes

## Stack Tecnológico

### Orquestación y LLM
- **LangChain**: Framework de orquestación
- **OpenAI GPT-3.5/4**: Modelo de lenguaje principal (probado)
- **Anthropic Claude**: Modelo alternativo (probado)
- **Sentence Transformers**: Embeddings locales

### Vector Store
- **ChromaDB**: Base de datos vectorial
- **FAISS**: Búsqueda de similitud (fallback)

### Interfaz y Backend
- **Streamlit**: Framework web interactivo
- **Plotly**: Visualizaciones interactivas
- **Pandas**: Análisis de datos

### Procesamiento
- **PyPDF2**: Extracción de texto de PDFs
- **scikit-learn**: Clustering y análisis
- **langdetect**: Detección de idioma

### Contenerización
- **Docker**: Contenerización de la aplicación
- **docker-compose**: Orquestación de servicios

## Instalación y Uso

### Opción 1: Docker (Recomendado)

```bash
# 1. Clonar repositorio
git clone https://github.com/NeoHowls/PDF_COPILOTO_OPENAI.git
cd PDF_COPILOTO_OPENAI

# 2. Configurar variables de entorno
start .env
# Editar .env con tus API keys(Solo probado con OPENAI)

# 3. Levantar servicios
docker-compose up --build

# 4. Abrir aplicación
# http://localhost:8501
```
## Configuración

### Variables de Entorno Clave

```bash
# APIs de LLM (al menos una - ambas probadas)
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

## Flujo Conversacional

### 1. Clasificación de Consultas
```python
QueryType = {
    "GENERAL_QUESTION",    # Pregunta general
    "DOCUMENT_SUMMARY",    # Solicitud de resumen
    "DOCUMENT_COMPARISON", # Comparación entre docs
    "SPECIFIC_SEARCH",     # Búsqueda específica
    "TOPIC_ANALYSIS"       # Análisis temático
}
```

### 2. Orquestación de Respuestas
```
Usuario → Clasificador → Procesador Específico → LLM → Respuesta
                ↓
        Vector Store ← Búsqueda Semántica
```

### 3. Contexto Conversacional
- **Historial de mensajes**
- **Documentos referenciados**
- **Chunks relevantes utilizados**
- **Temas discutidos**

## Funcionalidades de Análisis

### Resúmenes Automáticos
- Resumen por documento individual
- Resumen de colección completa
- Puntos clave extraídos automáticamente

### Comparación de Documentos
- Similitud de contenido (TF-IDF + Cosine)
- Solapamiento de temas (Jaccard)
- Análisis de diferencias usando LLM

### Clasificación Temática
- Clustering automático (K-means)
- Extracción de palabras clave
- Análisis de frecuencia de temas

### Métricas y Estadísticas
- Distribución de tamaños
- Análisis de similitudes
- Insights de colección

### Tests Implementados
- Procesamiento de PDFs
- Vector Store operations
- Orquestación de conversaciones
- Servicios de LLM (OpenAI y Anthropic)
- Análisis de documentos

### Métricas de Performance
- **Tiempo de procesamiento**: ~2-5s por PDF (dependiendo del tamaño)
- **Memoria RAM**: ~500MB base + ~100MB por documento
- **Vector Store**: ChromaDB optimizado para consultas rápidas
- **Concurrencia**: Soporte para múltiples usuarios (con limitaciones)

### Optimizaciones Implementadas
- Caching de embeddings
- Chunking inteligente con solapamiento
- Búsqueda híbrida optimizada
- Lazy loading de modelos
- Streaming de respuestas LLM

## Limitaciones Actuales
### Técnicas
- **Límite de archivos**: 5 PDFs máximo
- **Tamaño máximo**: 10MB por archivo
- **Tipos de archivo**: Solo PDFs con texto extraíble
- **Idiomas**: Optimizado para español e inglés
- **Concurrencia**: Una sesión por instancia

### Funcionales
- **PDFs con imágenes**: Texto en imágenes no se extrae
- **Tablas complejas**: Formato puede perderse
- **Documentos encriptados**: No soportados
- **Formatos mixtos**: Solo PDFs
### Errores
- **Iniciar sin API key**: Da mensaje de error de _type
- - **Solucion:** Iniciar con una API KEY

- **Lectura de PDF sin creditos**: Si se quedan sin creditos durante o antes de la lectura del PDF dara un error generico de OpenIA o posibilidad que la conversacion de rompa.
- - **Solucion:** Tener creditos en la cuenta o bien cambiar la API KEY


## Roadmap y Mejoras Futuras

### Corto Plazo
- **OCR Integration**: Extracción de texto de imágenes
- **Multi-idioma**: Soporte extendido para más idiomas
- **Autenticación**: Sistema de usuarios
- **Persistencia**: Sesiones guardadas
- **API REST**: Endpoints para integración

### Medio Plazo 
- **Múltiples Formatos**: Word, Excel, PowerPoint
- **Análisis de Imágenes**: Descripción automática
- **Grafos de Conocimiento**: Relaciones entre conceptos
- **Exportación**: Reportes en PDF/Word
- **Colaboración**: Sesiones compartidas

### Largo Plazo 
- **IA Multimodal**: Análisis de audio y video
- **Agentes Autónomos**: Tareas complejas automatizadas
- **Integración Cloud**: AWS/GCP/Azure
- **Escalabilidad**: Arquitectura distribuida
- **Personalización**: Modelos fine-tuned por dominio

## Consideraciones de Seguridad

### Datos y Privacidad
- **Archivos locales**: Los PDFs se procesan localmente
- **Vector Store**: Datos almacenados en contenedor local
- **API Keys**: Manejo seguro mediante variables de entorno
- **Sin persistencia cloud**: Los datos no salen del entorno local

### Recomendaciones
- Usar `.env` para API keys
- No incluir documentos sensibles en repos
- Limpiar datos regularmente
- Usar HTTPS en producción

## Documentación Adicional

### Estructura de Archivos
```
src/
├── config/          # Configuración centralizada
├── core/            # Lógica principal del negocio
├── models/          # Modelos de datos (Pydantic)
├── services/        # Servicios especializados
└── utils/           # Utilidades y helpers
```

### Modelos de Datos Clave
- **`Document`**: Representación completa de un PDF procesado
- **`TextChunk`**: Fragmento de texto con embeddings
- **`ConversationSession`**: Sesión de chat con contexto
- **`Message`**: Mensaje individual con metadatos

### Patrones de Diseño Utilizados
- **Factory Pattern**: `LLMService` para diferentes proveedores
- **Strategy Pattern**: Diferentes tipos de consulta
- **Observer Pattern**: Estado de sesión reactivo
- **Repository Pattern**: Acceso a datos del vector store

### Estándares de Código
- **Linting**: Black + flake8
- **Type Hints**: Obligatorios en funciones públicas
- **Docstrings**: Documentación clara y concisa
- **Tests**: Cobertura mínima del 80%

###Justificación de Elecciones Técnicas

Este proyecto fue desarrollado de manera individual, lo que hizo necesario tomar decisiones técnicas que garantizaran simplicidad, escalabilidad y facilidad de mantenimiento.

##Arquitectura y Diseño

Se implementó una arquitectura modular basada en capas (UI, Orquestación y Servicios). Esta decisión permite mantener el sistema escalable, con componentes desacoplados y fáciles de probar. La inyección de dependencias asegura que cada módulo pueda evolucionar de forma independiente.

##Vector Store

Se seleccionó ChromaDB por su balance entre rendimiento y facilidad de despliegue. Al funcionar localmente evita problemas de latencia, es sencillo de integrar con Docker y soporta metadata y filtrado avanzado. Además, su compatibilidad directa con Sentence Transformers lo convierte en la mejor opción frente a alternativas externas como Pinecone o Weaviate.

##Modelos de Lenguaje

La integración de OpenAI y Anthropic responde a la necesidad de combinar distintas fortalezas. Esta elección asegura continuidad del servicio, permite optimizar costos según la naturaleza de la consulta y aprovecha las capacidades complementarias de cada proveedor.

##Framework de Procesamiento

Se optó por LangChain como framework central porque abstrae la complejidad de trabajar con múltiples LLMs y ofrece herramientas específicas para RAG. Su comunidad activa y la integración nativa con embeddings y vector stores fueron factores clave.

##Interfaz de Usuario

Se utilizó Streamlit para construir una interfaz rápida de implementar y fácil de desplegar. Esta herramienta evita la necesidad de desarrollar un frontend complejo y aporta componentes listos (chat, carga de archivos, visualizaciones) que agilizan el desarrollo individual.

##Embeddings

Se eligieron Sentence Transformers debido a la calidad de sus embeddings, su soporte multilingüe y la posibilidad de ejecutarlos localmente. Esto permite mayor control y elimina la dependencia de servicios externos para el procesamiento semántico.
## Soporte y Contacto

### Issues y Bugs
- Reportar en GitHub Issues
- Incluir logs y pasos para reproducir
- Especificar versión y entorno


**CatchAI Copiloto** - Transforma tus documentos en conocimiento conversacional

*Desarrollado para el desafío técnico CatchAI*
