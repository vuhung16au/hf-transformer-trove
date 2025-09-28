#!/usr/bin/env python3
"""
vLLM Basic Usage Examples

This script demonstrates how to use vLLM for high-throughput inference
with hate speech detection models in an offline environment.
"""

import os
import time
import requests
import json
from typing import List, Dict, Optional
import subprocess
import torch

# Configuration for offline deployment
class vLLMConfig:
    """Configuration class for vLLM deployment."""
    
    # Preferred hate speech detection models (aligned with repository guidelines)
    PREFERRED_MODELS = [
        "cardiffnlp/twitter-roberta-base-hate-latest",
        "facebook/roberta-hate-speech-dynabench-r4-target", 
        "GroNLP/hateBERT",
        "Hate-speech-CNERG/dehatebert-mono-english",
    ]
    
    def __init__(self, model_name: str = None, offline: bool = True):
        self.model_name = model_name or self.PREFERRED_MODELS[0]
        self.offline = offline
        self.server_url = "http://localhost:8000"
        self.model_path = f"/models/{self.model_name.replace('/', '_')}"
        
    def get_device(self) -> str:
        """Get the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

class vLLMClient:
    """Client for interacting with vLLM server."""
    
    def __init__(self, config: vLLMConfig):
        self.config = config
        self.base_url = f"{config.server_url}/v1"
        
    def is_server_running(self) -> bool:
        """Check if vLLM server is running."""
        try:
            response = requests.get(f"{self.config.server_url}/health", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def wait_for_server(self, timeout: int = 60) -> bool:
        """Wait for server to be ready."""
        print("⏳ Waiting for vLLM server to start...")
        
        for i in range(timeout):
            if self.is_server_running():
                print("✅ vLLM server is ready!")
                return True
            time.sleep(1)
            
        print("❌ Server failed to start within timeout")
        return False
    
    def generate_text(self, 
                     prompt: str, 
                     max_tokens: int = 50,
                     temperature: float = 0.7) -> Dict:
        """Generate text using vLLM completion API."""
        
        payload = {
            "model": "hate-speech-detector",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            print(f"❌ Error generating text: {e}")
            return {"error": str(e)}
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]],
                       max_tokens: int = 50) -> Dict:
        """Use chat completion API for conversational interface."""
        
        payload = {
            "model": "hate-speech-detector", 
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,  # Lower temperature for hate speech detection
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
            
        except requests.RequestException as e:
            print(f"❌ Error in chat completion: {e}")
            return {"error": str(e)}

class vLLMServer:
    """Manager for vLLM server lifecycle."""
    
    def __init__(self, config: vLLMConfig):
        self.config = config
        self.process = None
        
    def start_server(self) -> bool:
        """Start vLLM server with Docker."""
        if self.is_docker_available():
            return self._start_with_docker()
        else:
            return self._start_with_python()
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            subprocess.run(["docker", "--version"], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _start_with_docker(self) -> bool:
        """Start server using Docker."""
        print("🐳 Starting vLLM server with Docker...")
        
        cmd = [
            "docker", "run", 
            "--rm",
            "--runtime", "nvidia" if self.config.get_device() == "cuda" else "",
            "--gpus", "all" if self.config.get_device() == "cuda" else "",
            "-v", f"{os.getcwd()}/models:/models",
            "-p", "8000:8000",
            "--name", "vllm-server",
            "-d",  # Detached mode
            "vllm-offline:latest",
            "--model", self.config.model_path,
            "--served-model-name", "hate-speech-detector",
            "--max-model-len", "2048",
            "--dtype", "half",
            "--trust-remote-code"
        ]
        
        # Remove empty strings from cmd
        cmd = [arg for arg in cmd if arg]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Docker container started: {result.stdout.strip()}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start Docker container: {e.stderr}")
            return False
    
    def _start_with_python(self) -> bool:
        """Start server using Python (fallback)."""
        print("🐍 Starting vLLM server with Python...")
        
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_name,
            "--served-model-name", "hate-speech-detector",
            "--max-model-len", "2048",
            "--dtype", "half" if self.config.get_device() == "cuda" else "float32"
        ]
        
        try:
            self.process = subprocess.Popen(cmd, 
                                          stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE)
            print("✅ Python server process started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start Python server: {e}")
            return False
    
    def stop_server(self):
        """Stop the vLLM server."""
        if self.is_docker_available():
            try:
                subprocess.run(["docker", "stop", "vllm-server"], 
                             capture_output=True, check=True)
                print("✅ Docker container stopped")
            except subprocess.CalledProcessError:
                print("⚠️  Docker container may not be running")
        
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("✅ Python server stopped")

def demonstrate_basic_usage():
    """Demonstrate basic vLLM usage patterns."""
    print("🚀 vLLM BASIC USAGE DEMONSTRATION")
    print("=" * 40)
    
    # Initialize configuration
    config = vLLMConfig(offline=True)
    print(f"📱 Using device: {config.get_device()}")
    print(f"🤖 Model: {config.model_name}")
    
    # Initialize server and client
    server = vLLMServer(config)
    client = vLLMClient(config)
    
    try:
        # Start server if not running
        if not client.is_server_running():
            print("🔄 Starting vLLM server...")
            if server.start_server():
                client.wait_for_server()
            else:
                print("❌ Failed to start server")
                return
        
        # Test hate speech detection examples
        print("\n🛡️  HATE SPEECH DETECTION EXAMPLES")
        print("-" * 35)
        
        test_texts = [
            "I love this community! Everyone is so supportive.",
            "This is a great example of positive communication.",
            "AI technology is advancing rapidly these days.",
            "I disagree with this policy but respect others' opinions."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n📝 Example {i}: '{text}'")
            
            # Generate classification
            result = client.generate_text(
                prompt=f"Classify this text for hate speech (safe/unsafe): {text}\nClassification:",
                max_tokens=10,
                temperature=0.1  # Low temperature for consistent classification
            )
            
            if "choices" in result:
                response = result["choices"][0]["text"].strip()
                print(f"🔍 Classification: {response}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            time.sleep(0.5)  # Rate limiting
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        server.stop_server()

def demonstrate_chat_interface():
    """Demonstrate chat-based hate speech moderation."""
    print("\n💬 CHAT-BASED MODERATION DEMO")
    print("=" * 35)
    
    config = vLLMConfig(offline=True)
    client = vLLMClient(config)
    
    if not client.is_server_running():
        print("❌ Server not running. Please start server first.")
        return
    
    # Simulate moderation conversation
    messages = [
        {
            "role": "system",
            "content": "You are a content moderation assistant. Classify messages as 'SAFE' or 'UNSAFE' and explain briefly."
        },
        {
            "role": "user", 
            "content": "Please review this message: 'I love learning about AI and machine learning!'"
        }
    ]
    
    result = client.chat_completion(messages, max_tokens=30)
    
    if "choices" in result:
        response = result["choices"][0]["message"]["content"]
        print(f"🤖 Moderation result: {response}")
    else:
        print(f"❌ Error: {result.get('error', 'Unknown error')}")

def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    print("\n📦 BATCH PROCESSING DEMO")
    print("=" * 30)
    
    config = vLLMConfig(offline=True)
    client = vLLMClient(config)
    
    if not client.is_server_running():
        print("❌ Server not running. Please start server first.")
        return
    
    # Batch of texts to process
    batch_texts = [
        "This is a wonderful day!",
        "I enjoy learning new things.",
        "Technology is fascinating.",
        "Community support is amazing.",
        "Positive feedback helps everyone grow."
    ]
    
    print(f"🔄 Processing batch of {len(batch_texts)} texts...")
    start_time = time.time()
    
    results = []
    for text in batch_texts:
        result = client.generate_text(
            prompt=f"Rate sentiment (positive/negative/neutral): {text}\nRating:",
            max_tokens=5,
            temperature=0.1
        )
        results.append(result)
    
    end_time = time.time()
    
    print(f"⏱️  Batch processing completed in {end_time - start_time:.2f}s")
    print(f"🚀 Average time per text: {(end_time - start_time) / len(batch_texts):.3f}s")
    
    # Display results
    for i, (text, result) in enumerate(zip(batch_texts, results), 1):
        if "choices" in result:
            sentiment = result["choices"][0]["text"].strip()
            print(f"{i}. '{text[:30]}...' → {sentiment}")

if __name__ == "__main__":
    """Main execution with educational examples."""
    print("📚 vLLM Educational Examples")
    print("🎯 Focus: Hate Speech Detection with High Throughput")
    print("=" * 50)
    
    try:
        # Basic usage demonstration
        demonstrate_basic_usage()
        
        # Wait before next demo
        input("\n⏭️  Press Enter to continue with chat interface demo...")
        demonstrate_chat_interface()
        
        # Wait before batch processing
        input("\n⏭️  Press Enter to continue with batch processing demo...")
        demonstrate_batch_processing()
        
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Make sure Docker is installed and models are downloaded")
    
    print("\n✅ vLLM basic usage examples completed!")
    print("🔗 Next: Try performance_comparison.ipynb for benchmarking")