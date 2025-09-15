# Usar una imagen base de Python 3.12 slim
FROM python:3.12-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del proyecto
COPY . .

# Instalar dependencias y descargar el modelo de spacy
RUN pip install --no-cache-dir . && \
    python -m spacy download en_core_web_sm

# Exponer el puerto que usará la aplicación
EXPOSE 8000

# Comando para iniciar la aplicación
CMD ["elysia", "start", "--host", "0.0.0.0", "--port", "8000"]
