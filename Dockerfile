# Use a slim Python 3.12 base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project into the image
COPY . .

# Install dependencies and download the small English spaCy model
RUN pip install --no-cache-dir . && \
    python -m spacy download en_core_web_sm

# Expose the port used by the app server
EXPOSE 8000

# Default command to run Elysia's API
CMD ["elysia", "start", "--host", "0.0.0.0", "--port", "8000"]
