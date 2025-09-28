#!/usr/bin/env python3
"""
llama.cpp Quantization Demo

This script demonstrates model quantization techniques for llama.cpp,
focusing on optimizing hate speech detection models for edge deployment.
"""

import os
import json
import time
import subprocess
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    
    # Source model information
    source_model: str = "cardiffnlp/twitter-roberta-base-hate-latest"
    model_type: str = "roberta"  # Model architecture type
    
    # Quantization settings
    quantization_types: List[str] = None
    output_dir: str = "./quantized_models"
    
    # Conversion settings
    context_size: int = 2048
    vocab_type: str = "bpe"
    
    def __post_init__(self):
        if self.quantization_types is None:
            self.quantization_types = ["q4_0", "q8_0", "q2_k", "q4_k_m", "q5_k_m"]

class ModelConverter:
    """Handle model conversion from HuggingFace to GGUF format."""
    
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.llama_cpp_path = self._find_llama_cpp_path()
        
    def _find_llama_cpp_path(self) -> Optional[str]:
        """Find llama.cpp installation path."""
        
        # Check common locations
        possible_paths = [
            "/llama.cpp",
            "./llama.cpp",
            os.path.expanduser("~/llama.cpp"),
            "/usr/local/bin/llama.cpp"
        ]
        
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "convert.py")):
                return path
        
        logger.warning("llama.cpp not found in common locations")
        return None
    
    def download_and_convert_model(self) -> Optional[str]:
        """Download HuggingFace model and convert to GGUF format."""
        
        if not self.llama_cpp_path:
            logger.error("llama.cpp not found. Please install llama.cpp first.")
            return None
        
        logger.info(f"🤖 Converting model: {self.config.source_model}")
        
        try:
            # Create output directory
            os.makedirs(self.config.output_dir, exist_ok=True)
            
            # Download model using transformers
            from transformers import AutoTokenizer, AutoModel
            
            model_local_path = os.path.join(
                self.config.output_dir, 
                self.config.source_model.replace("/", "_")
            )
            
            logger.info(f"📥 Downloading model to {model_local_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.config.source_model)
            model = AutoModel.from_pretrained(self.config.source_model)
            
            # Save locally
            tokenizer.save_pretrained(model_local_path)
            model.save_pretrained(model_local_path)
            
            # Convert to GGUF
            gguf_path = f"{model_local_path}.gguf"
            
            convert_cmd = [
                "python", os.path.join(self.llama_cpp_path, "convert.py"),
                model_local_path,
                "--outfile", gguf_path,
                "--outtype", "f16"
            ]
            
            logger.info(f"🔄 Converting to GGUF format...")
            
            result = subprocess.run(
                convert_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if os.path.exists(gguf_path):
                logger.info(f"✅ Model converted successfully: {gguf_path}")
                return gguf_path
            else:
                logger.error("❌ GGUF file not created")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Conversion failed: {e.stderr}")
            return None
        except ImportError:
            logger.error("❌ transformers library not installed")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error during conversion: {e}")
            return None

class ModelQuantizer:
    """Handle model quantization using llama.cpp tools."""
    
    def __init__(self, config: QuantizationConfig, llama_cpp_path: str):
        self.config = config
        self.llama_cpp_path = llama_cpp_path
        self.quantize_binary = os.path.join(llama_cpp_path, "quantize")
        
    def quantize_model(self, source_gguf_path: str) -> Dict[str, str]:
        """Quantize model to different precision levels."""
        
        if not os.path.exists(self.quantize_binary):
            logger.error(f"quantize binary not found at {self.quantize_binary}")
            return {}\
        
        results = {}\n        base_name = os.path.splitext(os.path.basename(source_gguf_path))[0]\n        \n        for quant_type in self.config.quantization_types:\n            logger.info(f\"🔄 Quantizing to {quant_type} format...\")\n            \n            output_path = os.path.join(\n                self.config.output_dir,\n                f\"{base_name}-{quant_type}.gguf\"\n            )\n            \n            try:\n                cmd = [\n                    self.quantize_binary,\n                    source_gguf_path,\n                    output_path,\n                    quant_type\n                ]\n                \n                start_time = time.time()\n                \n                result = subprocess.run(\n                    cmd,\n                    capture_output=True,\n                    text=True,\n                    check=True\n                )\n                \n                end_time = time.time()\n                \n                if os.path.exists(output_path):\n                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB\n                    \n                    logger.info(\n                        f\"✅ {quant_type}: {file_size:.1f}MB \"\n                        f\"(took {end_time - start_time:.1f}s)\"\n                    )\n                    \n                    results[quant_type] = {\n                        \"path\": output_path,\n                        \"size_mb\": file_size,\n                        \"quantization_time\": end_time - start_time\n                    }\n                else:\n                    logger.error(f\"❌ Failed to create {quant_type} quantized model\")\n                    \n            except subprocess.CalledProcessError as e:\n                logger.error(f\"❌ Quantization failed for {quant_type}: {e.stderr}\")\n            except Exception as e:\n                logger.error(f\"❌ Unexpected error quantizing {quant_type}: {e}\")\n        \n        return results

class PerformanceBenchmark:
    """Benchmark quantized models for performance comparison."""
    
    def __init__(self, llama_cpp_path: str):\n        self.llama_cpp_path = llama_cpp_path\n        self.server_binary = os.path.join(llama_cpp_path, \"server\")\n        self.current_server_process = None\n        \n    def start_server(self, model_path: str, port: int = 8080) -> bool:\n        \"\"\"Start llama.cpp server with specified model.\"\"\"\n        \n        if not os.path.exists(self.server_binary):\n            logger.error(f\"Server binary not found: {self.server_binary}\")\n            return False\n            \n        # Stop existing server\n        self.stop_server()\n        \n        cmd = [\n            self.server_binary,\n            \"--model\", model_path,\n            \"--host\", \"0.0.0.0\",\n            \"--port\", str(port),\n            \"--ctx-size\", \"2048\",\n            \"--threads\", str(os.cpu_count()),\n            \"--batch-size\", \"512\"\n        ]\n        \n        try:\n            logger.info(f\"🚀 Starting server with model: {os.path.basename(model_path)}\")\n            \n            self.current_server_process = subprocess.Popen(\n                cmd,\n                stdout=subprocess.PIPE,\n                stderr=subprocess.PIPE,\n                text=True\n            )\n            \n            # Wait for server to start\n            time.sleep(10)\n            \n            # Check if server is responsive\n            if self._is_server_ready(port):\n                logger.info(\"✅ Server started successfully\")\n                return True\n            else:\n                logger.error(\"❌ Server failed to start properly\")\n                self.stop_server()\n                return False\n                \n        except Exception as e:\n            logger.error(f\"❌ Failed to start server: {e}\")\n            return False\n    \n    def stop_server(self):\n        \"\"\"Stop the currently running server.\"\"\"\n        if self.current_server_process:\n            try:\n                self.current_server_process.terminate()\n                self.current_server_process.wait(timeout=10)\n                logger.info(\"✅ Server stopped\")\n            except subprocess.TimeoutExpired:\n                self.current_server_process.kill()\n                logger.warning(\"⚠️ Server killed (didn't terminate gracefully)\")\n            except Exception as e:\n                logger.error(f\"Error stopping server: {e}\")\n            \n            self.current_server_process = None\n    \n    def _is_server_ready(self, port: int, timeout: int = 30) -> bool:\n        \"\"\"Check if server is ready to accept requests.\"\"\"\n        \n        for _ in range(timeout):\n            try:\n                response = requests.get(f\"http://localhost:{port}/health\", timeout=1)\n                if response.status_code == 200:\n                    return True\n            except requests.RequestException:\n                pass\n            \n            time.sleep(1)\n        \n        return False\n    \n    def benchmark_model(self, model_path: str, test_texts: List[str], port: int = 8080) -> Dict:\n        \"\"\"Benchmark a specific quantized model.\"\"\"\n        \n        if not self.start_server(model_path, port):\n            return {\"error\": \"Failed to start server\"}\n        \n        try:\n            results = {\n                \"model\": os.path.basename(model_path),\n                \"model_size_mb\": os.path.getsize(model_path) / (1024 * 1024),\n                \"response_times\": [],\n                \"throughput\": 0,\n                \"avg_response_time\": 0,\n                \"errors\": 0\n            }\n            \n            logger.info(f\"📊 Benchmarking {len(test_texts)} requests...\")\n            \n            start_time = time.time()\n            \n            for i, text in enumerate(test_texts, 1):\n                try:\n                    # Make inference request\n                    payload = {\n                        \"prompt\": f\"Classify this text sentiment: {text}\\nClassification:\",\n                        \"n_predict\": 10,\n                        \"temperature\": 0.1,\n                        \"stop\": [\"\\n\"]\n                    }\n                    \n                    request_start = time.time()\n                    \n                    response = requests.post(\n                        f\"http://localhost:{port}/completion\",\n                        json=payload,\n                        timeout=30\n                    )\n                    \n                    request_end = time.time()\n                    response_time = request_end - request_start\n                    \n                    if response.status_code == 200:\n                        results[\"response_times\"].append(response_time)\n                        \n                        if i % 5 == 0:  # Progress update\n                            logger.info(f\"  Progress: {i}/{len(test_texts)} ({response_time:.2f}s)\")\n                    else:\n                        results[\"errors\"] += 1\n                        logger.warning(f\"Request {i} failed: {response.status_code}\")\n                        \n                except requests.RequestException as e:\n                    results[\"errors\"] += 1\n                    logger.warning(f\"Request {i} error: {e}\")\n                \n                # Small delay between requests\n                time.sleep(0.1)\n            \n            end_time = time.time()\n            total_time = end_time - start_time\n            \n            # Calculate metrics\n            if results[\"response_times\"]:\n                results[\"avg_response_time\"] = sum(results[\"response_times\"]) / len(results[\"response_times\"])\n                results[\"throughput\"] = len(results[\"response_times\"]) / total_time\n                results[\"success_rate\"] = len(results[\"response_times\"]) / len(test_texts) * 100\n            \n            logger.info(\n                f\"✅ Benchmark completed: \"\n                f\"avg={results['avg_response_time']:.3f}s, \"\n                f\"throughput={results['throughput']:.2f} req/s, \"\n                f\"success={results.get('success_rate', 0):.1f}%\"\n            )\n            \n            return results\n            \n        finally:\n            self.stop_server()\n    \n    def compare_quantizations(self, quantized_models: Dict[str, str], test_texts: List[str]) -> Dict:\n        \"\"\"Compare performance across different quantization levels.\"\"\"\n        \n        comparison_results = {}\n        \n        logger.info(f\"🏁 Starting quantization comparison...\")\n        \n        for quant_type, model_info in quantized_models.items():\n            model_path = model_info[\"path\"]\n            \n            logger.info(f\"\\n📊 Benchmarking {quant_type} quantization...\")\n            \n            benchmark_result = self.benchmark_model(model_path, test_texts)\n            \n            if \"error\" not in benchmark_result:\n                comparison_results[quant_type] = {\n                    **benchmark_result,\n                    \"quantization_info\": model_info\n                }\n            else:\n                logger.error(f\"❌ Benchmark failed for {quant_type}: {benchmark_result['error']}\")\n        \n        return comparison_results

def create_test_dataset() -> List[str]:\n    \"\"\"Create test dataset for hate speech detection benchmarking.\"\"\"\n    \n    # Focus on educational and positive examples aligned with repository guidelines\n    test_texts = [\n        \"I love learning about artificial intelligence and machine learning technologies.\",\n        \"This educational content is very helpful for understanding transformers.\",\n        \"The community here is supportive and encourages collaborative learning.\",\n        \"Great explanation of the technical concepts, very clear and informative.\",\n        \"I appreciate the detailed documentation and examples provided.\",\n        \"Constructive feedback helps improve the quality of our work.\",\n        \"The research findings are interesting and well-presented.\",\n        \"Thank you for sharing your knowledge and expertise with others.\",\n        \"This approach is innovative and shows promising results.\",\n        \"Educational resources like this make complex topics more accessible.\",\n        \"I enjoy participating in technical discussions and learning from others.\",\n        \"The implementation is efficient and follows best practices.\",\n        \"Open source contributions benefit the entire community.\",\n        \"The tutorial is comprehensive and easy to follow step by step.\",\n        \"Collaboration between researchers leads to better outcomes.\"\n    ]\n    \n    return test_texts

def generate_performance_report(comparison_results: Dict) -> str:\n    \"\"\"Generate a comprehensive performance comparison report.\"\"\"\n    \n    if not comparison_results:\n        return \"No benchmark results available.\"\n    \n    report = [\"\\n📊 QUANTIZATION PERFORMANCE REPORT\", \"=\" * 40, \"\"]\n    \n    # Summary table\n    report.append(\"📋 Summary Table:\")\n    report.append(\n        f\"{'Quantization':<12} {'Size(MB)':<10} {'Avg Time(s)':<12} \"\n        f\"{'Throughput':<12} {'Success%':<10}\"\n    )\n    report.append(\"-\" * 65)\n    \n    for quant_type, results in comparison_results.items():\n        report.append(\n            f\"{quant_type:<12} {results['model_size_mb']:<10.1f} \"\n            f\"{results['avg_response_time']:<12.3f} \"\n            f\"{results['throughput']:<12.2f} \"\n            f\"{results.get('success_rate', 0):<10.1f}\"\n        )\n    \n    # Detailed analysis\n    report.append(\"\\n🔍 Detailed Analysis:\")\n    \n    # Find best performers\n    if comparison_results:\n        fastest_quant = min(\n            comparison_results.items(),\n            key=lambda x: x[1]['avg_response_time']\n        )\n        \n        highest_throughput = max(\n            comparison_results.items(),\n            key=lambda x: x[1]['throughput']\n        )\n        \n        smallest_size = min(\n            comparison_results.items(),\n            key=lambda x: x[1]['model_size_mb']\n        )\n        \n        report.extend([\n            f\"⚡ Fastest Response: {fastest_quant[0]} ({fastest_quant[1]['avg_response_time']:.3f}s)\",\n            f\"🚀 Highest Throughput: {highest_throughput[0]} ({highest_throughput[1]['throughput']:.2f} req/s)\",\n            f\"💾 Smallest Size: {smallest_size[0]} ({smallest_size[1]['model_size_mb']:.1f}MB)\"\n        ])\n    \n    # Recommendations\n    report.extend([\n        \"\\n💡 Recommendations:\",\n        \"   • q4_0: Good balance of quality and performance\",\n        \"   • q8_0: Higher quality, larger size\",\n        \"   • q2_k: Smallest size, acceptable quality for edge devices\",\n        \"   • q4_k_m/q5_k_m: Advanced quantization with better quality retention\"\n    ])\n    \n    return \"\\n\".join(report)

def demonstrate_quantization_workflow():\n    \"\"\"Demonstrate the complete quantization workflow.\"\"\"\n    \n    print(\"🔧 LLAMA.CPP QUANTIZATION DEMO\")\n    print(\"=\" * 35)\n    print(\"🎯 Focus: Hate Speech Detection Model Optimization\")\n    print(\"💻 Target: CPU-optimized inference for edge deployment\\n\")\n    \n    # Initialize configuration\n    config = QuantizationConfig(\n        source_model=\"cardiffnlp/twitter-roberta-base-hate-latest\",\n        quantization_types=[\"q4_0\", \"q8_0\", \"q2_k\"],  # Subset for demo\n        output_dir=\"./quantized_models\"\n    )\n    \n    print(f\"📋 Configuration:\")\n    print(f\"   Source Model: {config.source_model}\")\n    print(f\"   Quantization Types: {', '.join(config.quantization_types)}\")\n    print(f\"   Output Directory: {config.output_dir}\")\n    \n    try:\n        # Step 1: Convert model\n        print(f\"\\n🔄 Step 1: Model Conversion\")\n        converter = ModelConverter(config)\n        \n        if not converter.llama_cpp_path:\n            print(\"❌ llama.cpp not found. Please install it first.\")\n            print(\"💡 Run: git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make\")\n            return\n        \n        gguf_model_path = converter.download_and_convert_model()\n        \n        if not gguf_model_path:\n            print(\"❌ Model conversion failed\")\n            return\n        \n        # Step 2: Quantize model\n        print(f\"\\n🔄 Step 2: Model Quantization\")\n        quantizer = ModelQuantizer(config, converter.llama_cpp_path)\n        quantized_models = quantizer.quantize_model(gguf_model_path)\n        \n        if not quantized_models:\n            print(\"❌ Model quantization failed\")\n            return\n        \n        print(f\"\\n✅ Quantized models created:\")\n        for quant_type, info in quantized_models.items():\n            print(f\"   {quant_type}: {info['size_mb']:.1f}MB\")\n        \n        # Step 3: Performance benchmarking\n        print(f\"\\n🔄 Step 3: Performance Benchmarking\")\n        \n        benchmark = PerformanceBenchmark(converter.llama_cpp_path)\n        test_texts = create_test_dataset()[:10]  # Use subset for demo\n        \n        print(f\"📊 Running benchmarks on {len(test_texts)} test examples...\")\n        \n        comparison_results = benchmark.compare_quantizations(quantized_models, test_texts)\n        \n        # Step 4: Generate report\n        if comparison_results:\n            report = generate_performance_report(comparison_results)\n            print(report)\n            \n            # Save report\n            with open(\"quantization_report.txt\", \"w\") as f:\n                f.write(report)\n            print(f\"\\n💾 Report saved to quantization_report.txt\")\n        \n        else:\n            print(\"❌ Benchmarking failed\")\n        \n        print(f\"\\n✅ Quantization demo completed!\")\n        print(f\"💡 Next steps:\")\n        print(f\"   • Test quantized models in production\")\n        print(f\"   • Deploy to edge devices\")\n        print(f\"   • Monitor performance in real-world scenarios\")\n        \n    except KeyboardInterrupt:\n        print(\"\\n⏹️ Demo interrupted by user\")\n    except Exception as e:\n        logger.error(f\"❌ Demo failed: {e}\")\n    finally:\n        # Cleanup\n        print(\"\\n🧹 Cleaning up...\")\n        # Clean up any running processes\n        try:\n            subprocess.run([\"pkill\", \"-f\", \"llama-server\"], capture_output=True)\n        except:\n            pass

if __name__ == \"__main__\":\n    \"\"\"Main execution with quantization demonstration.\"\"\"\n    \n    print(\"📚 llama.cpp Quantization Educational Demo\")\n    print(\"🎯 Learning: Model Optimization for Edge Deployment\")\n    print(\"=\" * 55)\n    \n    try:\n        demonstrate_quantization_workflow()\n        \n    except KeyboardInterrupt:\n        print(\"\\n👋 Goodbye!\")\n    except Exception as e:\n        print(f\"\\n❌ Unexpected error: {e}\")\n        print(\"💡 Make sure llama.cpp is installed and accessible\")\n    \n    print(\"\\n🔗 Next: Try cpu_inference.ipynb for detailed CPU performance analysis\")