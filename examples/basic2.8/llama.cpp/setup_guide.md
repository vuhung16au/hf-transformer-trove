# llama.cpp Setup Guide

## 🎯 Overview

llama.cpp is a high-performance CPU-optimized inference engine for LLMs, designed for edge deployment and resource-constrained environments. This guide shows you how to set up llama.cpp for offline deployment using Docker.

## 🏃‍♂️ Key Features

- **CPU Optimization**: Highly optimized for CPU inference with SIMD instructions
- **Quantization**: Advanced model quantization (4-bit, 8-bit, mixed precision)
- **Memory Efficiency**: Minimal memory footprint for edge devices
- **Cross-Platform**: Works on ARM, x86, and various architectures
- **No Dependencies**: Minimal external dependencies for easy deployment

## 🐳 Docker Setup (Offline Deployment)

### Prerequisites

- Docker installed and running
- At least 4GB RAM for smaller models
- Models pre-converted to GGUF format

### 1. Build llama.cpp Docker Image

Create the Docker image for offline deployment:

```bash
# Clone llama.cpp repository
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build Docker image with all optimizations
docker build -t llamacpp-offline:latest .
```

### 2. Alternative: Use Pre-built Image

Build a custom optimized image:

```dockerfile
FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Clone and build llama.cpp
RUN git clone https://github.com/ggerganov/llama.cpp.git /llama.cpp
WORKDIR /llama.cpp

# Build with optimizations
RUN make -j$(nproc) LLAMA_NATIVE=1

# Create model directory
RUN mkdir -p /models

EXPOSE 8080

CMD ["./server", "--host", "0.0.0.0", "--port", "8080"]
```

### 3. Create Offline Deployment Script

Create a script to run llama.cpp offline:

```bash
#!/bin/bash
# run_llamacpp_offline.sh

# Set model path (models should be pre-converted to GGUF)
MODEL_PATH="/models/hate-speech-model.gguf"

# Run llama.cpp server
docker run --rm \
    -v $(pwd)/models:/models \
    -p 8080:8080 \
    --name llamacpp-server \
    llamacpp-offline:latest \
    ./server \
    --host 0.0.0.0 \
    --port 8080 \
    --model $MODEL_PATH \
    --ctx-size 2048 \
    --threads $(nproc) \
    --n-gpu-layers 0 \
    --verbose
```

## 📦 Model Preparation for Offline Use

### 1. Convert Models to GGUF Format

llama.cpp requires models in GGUF format. Convert your models:

```python
# convert_to_gguf.py
import os
import subprocess
from transformers import AutoTokenizer, AutoModel

def convert_hf_to_gguf(model_name: str, output_dir: str = "./models/"):
    """Convert HuggingFace model to GGUF format."""
    
    print(f"📥 Converting {model_name} to GGUF format...")
    
    # Download model first
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Save locally
    local_path = os.path.join(output_dir, model_name.replace("/", "_"))
    os.makedirs(local_path, exist_ok=True)
    
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    
    # Convert using llama.cpp converter
    gguf_path = f"{local_path}.gguf"
    
    try:
        # Note: This is a simplified example
        # Actual conversion may require model-specific scripts
        convert_cmd = [
            "python", "/llama.cpp/convert.py",
            local_path,
            "--outfile", gguf_path,
            "--outtype", "f16"
        ]
        
        subprocess.run(convert_cmd, check=True)
        print(f"✅ Model converted to {gguf_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Conversion failed: {e}")
        print("💡 Some models may require manual conversion")

# Convert preferred hate speech models
MODELS_TO_CONVERT = [
    "cardiffnlp/twitter-roberta-base-hate-latest",
    "microsoft/DialoGPT-small",  # Alternative for chat-like inference
]

for model in MODELS_TO_CONVERT:
    convert_hf_to_gguf(model)
```

### 2. Quantize Models for Efficiency

Create quantized versions for different use cases:

```bash
#!/bin/bash
# quantize_models.sh

MODEL_PATH="/models/hate-speech-model.gguf"

# Create different quantization levels
echo "🔄 Creating quantized model variants..."

# Q4_0: 4-bit quantization (good balance)
./quantize $MODEL_PATH ${MODEL_PATH%.gguf}-q4_0.gguf q4_0

# Q8_0: 8-bit quantization (higher quality)
./quantize $MODEL_PATH ${MODEL_PATH%.gguf}-q8_0.gguf q8_0

# Q2_K: 2-bit quantization (smallest size)
./quantize $MODEL_PATH ${MODEL_PATH%.gguf}-q2_k.gguf q2_k

echo "✅ Quantized models created"
echo "💡 Use different quantization levels based on your quality/size requirements"
```

## 🚀 Quick Start Commands

### Start llama.cpp Server (Offline)

```bash
# Make script executable
chmod +x run_llamacpp_offline.sh

# Start server
./run_llamacpp_offline.sh
```

### Test the Server

```bash
# Test with curl - completion endpoint
curl http://localhost:8080/completion \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Classify this text for hate speech: This is a positive message about learning AI.",
        "n_predict": 50,
        "temperature": 0.1,
        "stop": ["\n"]
    }'
```

### Chat Interface Testing

```bash
# Test chat endpoint
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hate-speech-detector",
        "messages": [
            {"role": "system", "content": "You are a content moderation assistant."},
            {"role": "user", "content": "Is this hate speech: I love learning about AI!"}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }'
```

## 🔧 Performance Tuning

### CPU Optimization

```bash
# Optimize for your CPU
docker run --rm \
    -v $(pwd)/models:/models \
    -p 8080:8080 \
    llamacpp-offline:latest \
    ./server \
    --model /models/model-q4_0.gguf \
    --threads $(nproc) \
    --ctx-size 2048 \
    --batch-size 512 \
    --mlock \
    --no-mmap
```

### Memory Optimization

```bash
# For limited RAM environments
docker run --rm \
    -v $(pwd)/models:/models \
    -p 8080:8080 \
    --memory=2g \
    llamacpp-offline:latest \
    ./server \
    --model /models/model-q2_k.gguf \
    --threads 2 \
    --ctx-size 1024 \
    --batch-size 256 \
    --low-vram
```

### GPU Acceleration (Optional)

```bash
# If GPU support is available
docker run --runtime nvidia --gpus all \
    -v $(pwd)/models:/models \
    -p 8080:8080 \
    llamacpp-offline:latest \
    ./server \
    --model /models/model.gguf \
    --n-gpu-layers 32 \
    --threads 4
```

## 📊 Monitoring and Health Checks

### Health Check Endpoint

```bash
# Check server health
curl http://localhost:8080/health
```

### Performance Metrics

```bash
# Get server stats
curl http://localhost:8080/stats
```

### Model Information

```bash
# Get model info
curl http://localhost:8080/props
```

## 🎯 Edge Deployment Considerations

### ARM Architecture Support

```bash
# For ARM devices (Raspberry Pi, etc.)
docker buildx build --platform linux/arm64 -t llamacpp-arm:latest .
```

### Resource Constraints

```yaml
# docker-compose.yml for edge deployment
version: '3.8'
services:
  llamacpp:
    image: llamacpp-offline:latest
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models:ro
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
    environment:
      - MODEL_PATH=/models/model-q4_0.gguf
      - THREADS=2
      - CTX_SIZE=1024
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## ⚠️ Troubleshooting

### Common Issues

1. **Model Loading Fails**
   ```bash
   # Check model format and path
   docker run -v $(pwd)/models:/models llamacpp-offline:latest ls -la /models
   ```

2. **Out of Memory**
   ```bash
   # Use smaller quantized model
   --model /models/model-q2_k.gguf
   --ctx-size 512
   ```

3. **Slow Performance**
   ```bash
   # Optimize thread usage
   --threads $(nproc)
   --batch-size 512
   --mlock
   ```

4. **Docker Build Issues**
   ```bash
   # Build with specific optimizations
   docker build --build-arg LLAMA_NATIVE=1 -t llamacpp-offline:latest .
   ```

## 📈 Performance Benchmarking

### Simple Benchmark Script

```bash
#!/bin/bash
# benchmark_llamacpp.sh

echo "🏁 Running llama.cpp performance benchmark..."

# Test different configurations
CONFIGS=(
    "--threads 1 --batch-size 256"
    "--threads $(nproc) --batch-size 512" 
    "--threads $(nproc) --batch-size 1024"
)

for config in "${CONFIGS[@]}"; do
    echo "Testing configuration: $config"
    
    # Start server with config
    timeout 60s docker run --rm -d \
        --name bench-test \
        -v $(pwd)/models:/models \
        -p 8080:8080 \
        llamacpp-offline:latest \
        ./server --model /models/model-q4_0.gguf $config
    
    sleep 10  # Wait for startup
    
    # Run benchmark requests
    for i in {1..10}; do
        curl -s http://localhost:8080/completion \
            -H "Content-Type: application/json" \
            -d '{
                "prompt": "Test prompt for benchmarking",
                "n_predict": 50,
                "temperature": 0.1
            }' > /dev/null
    done
    
    # Stop server
    docker stop bench-test
    sleep 2
done

echo "✅ Benchmark completed"
```

## 🔗 Next Steps

- Try `quantization_demo.py` for model optimization examples
- Run `cpu_inference.ipynb` for detailed CPU performance analysis
- Explore `edge_deployment.py` for IoT and edge device patterns

## 📚 References

- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
- [Quantization Guide](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)