# Usa la imagen base de Ollama
FROM ollama/ollama:latest

# Define el usuario como root para operaciones administrativas
USER root

# Crear el directorio /.ollama/
RUN mkdir -p /.ollama/

# Cambiar permisos del directorio para accesibilidad
RUN chmod -R 777 /.ollama/

# Actualizar los repositorios y agregar las herramientas necesarias
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3 \
    python-is-python3 \
    espeak-ng \
    espeak \
    ffmpeg \
    wget \
    && apt-get clean

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos de la aplicación al contenedor
COPY . /app
COPY app.py /app

# Instalar las dependencias de Python especificadas
RUN pip install --no-cache-dir -r requirements.txt
