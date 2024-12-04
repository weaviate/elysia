#!/bin/bash
set -e

# Start the FastAPI backend on localhost (not exposed publicly)
echo "Starting FastAPI backend..."
python elysia/api/cli.py start --port 8000 --host 0.0.0.0

# Start the Next.js frontend on the public interface
echo "Starting Next.js frontend..."
cd frontend
npm run start -- -p 3000

# Keep the script running
wait