FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip y setuptools
RUN pip install --upgrade pip setuptools wheel

# Establecer directorio de trabajo
WORKDIR /app

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY data/ ./data/

# Crear directorios necesarios
RUN mkdir -p /app/data/uploads /app/data/vectorstore

# Establecer variables de entorno
ENV PYTHONPATH=/app/src
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Exponer puerto
EXPOSE 8501

# Comando para ejecutar la aplicación
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
