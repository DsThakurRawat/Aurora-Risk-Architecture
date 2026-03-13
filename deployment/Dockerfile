FROM python:3.10-slim

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install dependencies (skipping torch/sklearn optimization for this demo)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY main.py .

# Expose FastAPI port
EXPOSE 8000

# Set Python path to resolve src imports
ENV PYTHONPATH=/app

# Command to run the FastAPI server via Uvicorn
CMD [\"uvicorn\", \"src.serving.api:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\", \"--workers\", \"4\"]
