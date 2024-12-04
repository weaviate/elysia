# Use an official Python runtime as a base image
FROM python:3.11

# Install Node.js
RUN apt-get update && \
    apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy backend files
COPY elysia/ ./elysia/

# Copy frontend files
COPY frontend/ ./frontend/

COPY setup.py ./
COPY requirements.txt ./

# Copy env file if it exists at root level
COPY .env* ./

# Install backend dependencies
RUN pip install --no-cache-dir '.'

# Install frontend dependencies and build
RUN cd frontend && \
    npm install && \
    npm run build

# Expose the port 3000 to the outside
EXPOSE 3000

# Copy the start script
COPY start_docker.sh /start_docker.sh
RUN chmod +x /start_docker.sh

# Start both servers
CMD ["/start_docker.sh"]