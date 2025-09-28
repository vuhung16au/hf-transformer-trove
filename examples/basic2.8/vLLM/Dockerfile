# Use official vLLM base image for offline deployment
FROM vllm/vllm-openai:latest

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV MODEL_PATH=/models
ENV SERVER_PORT=8000

# Install additional dependencies for production
RUN pip install --no-cache-dir \
    pyyaml \
    requests \
    psutil \
    prometheus-client

# Create necessary directories
RUN mkdir -p /models /logs /app

# Copy application scripts
COPY basic_usage.py /app/
COPY deployment_examples.py /app/

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN groupadd -r vllm && useradd -r -g vllm vllm
RUN chown -R vllm:vllm /app /logs

# Expose ports
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SERVER_PORT}/health || exit 1

# Default command with configurable options
CMD ["python", "-m", "vllm.entrypoints.openai.api_server", \
     "--host", "0.0.0.0", \
     "--port", "${SERVER_PORT}", \
     "--model", "${MODEL_PATH}", \
     "--served-model-name", "hate-speech-detector", \
     "--max-model-len", "2048", \
     "--dtype", "half", \
     "--trust-remote-code"]