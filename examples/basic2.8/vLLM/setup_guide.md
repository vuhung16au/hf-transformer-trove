# vLLM Setup Guide

## 🎯 Overview

vLLM is a high-throughput and memory-efficient inference engine for LLMs, optimized for serving large language models with maximum throughput. This guide shows you how to set up vLLM for offline deployment using Docker.

## ⚡ Key Features

- **PagedAttention**: Revolutionary memory management for attention computation
- **Continuous Batching**: Dynamic batching for optimal throughput
- **High Throughput**: Up to 24x higher throughput than baseline HF implementations
- **OpenAI Compatible API**: Drop-in replacement for OpenAI API
- **Multi-GPU Support**: Tensor and pipeline parallelism

## 🐳 Docker Setup (Offline Deployment)

### Prerequisites

- Docker installed and running
- NVIDIA Container Toolkit (for GPU support)
- At least 8GB VRAM for smaller models (16GB+ recommended)

### 1. Build vLLM Docker Image

Create the Docker image for offline deployment:

```bash
# Clone vLLM repository
git clone https://github.com/vllm-project/vllm.git
cd vllm

# Build Docker image with all dependencies
docker build -t vllm-offline:latest -f Dockerfile .
```

### 2. Alternative: Use Pre-built Image

If you have internet access initially to pull the image:

```bash
# Pull official vLLM image
docker pull vllm/vllm-openai:latest

# Tag it for offline use
docker tag vllm/vllm-openai:latest vllm-offline:latest
```

### 3. Create Offline Deployment Script

Create a script to run vLLM offline:

```bash
#!/bin/bash
# run_vllm_offline.sh

# Set model path (models should be pre-downloaded)
MODEL_PATH="/models/hate-speech-model"

# Run vLLM server
docker run --runtime nvidia --gpus all \
    -v $(pwd)/models:/models \
    -p 8000:8000 \
    --ipc=host \
    vllm-offline:latest \
    --model $MODEL_PATH \
    --port 8000 \
    --served-model-name hate-speech-detector \
    --max-model-len 2048 \
    --dtype half \
    --trust-remote-code
```

## 📦 Model Preparation for Offline Use

### 1. Download Preferred Models

Pre-download hate speech detection models for offline use:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# Preferred models for hate speech detection
MODELS = [
    "cardiffnlp/twitter-roberta-base-hate-latest",
    "facebook/roberta-hate-speech-dynabench-r4-target",
    "GroNLP/hateBERT",
]

def download_model(model_name, save_path):
    """Download and save model for offline use."""
    print(f"📥 Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save locally
    model_save_path = os.path.join(save_path, model_name.replace("/", "_"))
    os.makedirs(model_save_path, exist_ok=True)
    
    tokenizer.save_pretrained(model_save_path)
    model.save_pretrained(model_save_path)
    
    print(f"✅ Model saved to {model_save_path}")

# Download models
for model in MODELS:
    download_model(model, "./models/")
```

### 2. Create Model Configuration

Create a configuration for offline deployment:

```yaml
# vllm_config.yaml
models:
  - name: hate-speech-detector
    model: /models/cardiffnlp_twitter-roberta-base-hate-latest
    trust_remote_code: true
    max_model_len: 2048
    dtype: half
    
server:
  host: 0.0.0.0
  port: 8000
  max_num_batched_tokens: 2048
  max_num_seqs: 64
```

## 🚀 Quick Start Commands

### Start vLLM Server (Offline)

```bash
# Make script executable
chmod +x run_vllm_offline.sh

# Start server
./run_vllm_offline.sh
```

### Test the Server

```bash
# Test with curl
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "hate-speech-detector",
        "prompt": "This message contains harmful content:",
        "max_tokens": 50
    }'
```

## 🔧 Performance Tuning

### GPU Memory Optimization

```bash
# For limited VRAM
docker run --runtime nvidia --gpus all \
    -v $(pwd)/models:/models \
    -p 8000:8000 \
    vllm-offline:latest \
    --model /models/hate-speech-model \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.8
```

### CPU-Only Deployment

```bash
# For CPU-only environments
docker run \
    -v $(pwd)/models:/models \
    -p 8000:8000 \
    vllm-offline:latest \
    --model /models/hate-speech-model \
    --device cpu \
    --dtype float32
```

## 📊 Monitoring and Health Checks

### Health Check Endpoint

```bash
# Check server health
curl http://localhost:8000/health
```

### Metrics Collection

```bash
# Get server metrics
curl http://localhost:8000/metrics
```

## ⚠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   ```bash
   # Reduce memory usage
   --max-num-batched-tokens 512
   --gpu-memory-utilization 0.7
   ```

2. **Model Loading Fails**
   ```bash
   # Check model path and permissions
   docker run -v $(pwd)/models:/models vllm-offline:latest ls -la /models
   ```

3. **Slow Performance**
   ```bash
   # Enable optimizations
   --enable-prefix-caching
   --disable-log-stats
   ```

## 🔗 Next Steps

- Try `basic_usage.py` for simple examples
- Run `performance_comparison.ipynb` for benchmarking
- Explore `deployment_examples.py` for production patterns

## 📚 References

- [vLLM Documentation](https://docs.vllm.ai/)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [Model Hub Integration](https://huggingface.co/docs/transformers/model_doc/vllm)