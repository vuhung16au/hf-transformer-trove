#!/usr/bin/env python3
"""
vLLM Production Deployment Examples

This script demonstrates production-ready deployment patterns for vLLM
in offline Docker environments with comprehensive monitoring and scaling.
"""

import os
import json
import time
import logging
import threading
import subprocess
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import requests
import yaml
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vllm_deployment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for vLLM production deployment."""
    
    # Model configuration
    model_name: str = "cardiffnlp/twitter-roberta-base-hate-latest"
    model_path: str = "/models/cardiffnlp_twitter-roberta-base-hate-latest"
    served_model_name: str = "hate-speech-detector"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    max_model_len: int = 2048
    dtype: str = "half"
    
    # Performance tuning
    tensor_parallel_size: int = 1
    max_num_batched_tokens: int = 2048
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.9
    
    # Docker configuration
    container_name: str = "vllm-production"
    docker_image: str = "vllm-offline:latest"
    restart_policy: str = "unless-stopped"
    
    # Health check configuration
    health_check_interval: int = 30
    health_check_timeout: int = 10
    health_check_retries: int = 3
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 8080

class DockerDeploymentManager:
    """Manager for Docker-based vLLM deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.container_id: Optional[str] = None
        self.is_running = False
        self._shutdown_event = threading.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()
    
    def is_docker_available(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "info"], 
                capture_output=True, 
                check=True,
                timeout=10
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def build_docker_command(self) -> List[str]:
        """Build Docker command for vLLM deployment."""
        cmd = [
            "docker", "run",
            "--name", self.config.container_name,
            "--restart", self.config.restart_policy,
            "-d",  # Detached mode
            "-p", f"{self.config.port}:{self.config.port}",
            "-v", f"{os.getcwd()}/models:/models:ro",  # Read-only model volume
            "-v", f"{os.getcwd()}/logs:/logs",  # Logs volume
        ]
        
        # Add GPU support if available
        if self._is_nvidia_docker_available():
            cmd.extend(["--runtime", "nvidia", "--gpus", "all"])
        
        # Add environment variables
        cmd.extend([
            "-e", f"CUDA_VISIBLE_DEVICES=0",
            "-e", f"LOG_LEVEL={self.config.log_level}",
        ])
        
        # Add health check
        cmd.extend([
            "--health-cmd", f"curl -f http://localhost:{self.config.port}/health || exit 1",
            "--health-interval", f"{self.config.health_check_interval}s",
            "--health-timeout", f"{self.config.health_check_timeout}s",
            "--health-retries", str(self.config.health_check_retries),
        ])
        
        # Add the Docker image and vLLM arguments
        cmd.append(self.config.docker_image)
        
        # vLLM server arguments
        vllm_args = [
            "--model", self.config.model_path,
            "--served-model-name", self.config.served_model_name,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--max-model-len", str(self.config.max_model_len),
            "--dtype", self.config.dtype,
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--max-num-batched-tokens", str(self.config.max_num_batched_tokens),
            "--max-num-seqs", str(self.config.max_num_seqs),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--trust-remote-code",
        ]
        
        if self.config.enable_metrics:
            vllm_args.extend(["--enable-metrics"])
        
        cmd.extend(vllm_args)
        
        return cmd
    
    def _is_nvidia_docker_available(self) -> bool:
        """Check if NVIDIA Docker runtime is available."""
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.Runtimes}}"],
                capture_output=True,
                text=True,
                check=True
            )
            return "nvidia" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def deploy(self) -> bool:
        """Deploy vLLM service with Docker."""
        if not self.is_docker_available():
            logger.error("Docker is not available. Please install Docker and try again.")
            return False
        
        # Stop existing container if running
        self.stop()
        
        logger.info("🚀 Starting vLLM production deployment...")
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Build and execute Docker command
        cmd = self.build_docker_command()
        
        try:
            logger.info(f"Executing Docker command: {' '.join(cmd[:10])}...")  # Log truncated command
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.container_id = result.stdout.strip()
            
            logger.info(f"✅ Container started with ID: {self.container_id[:12]}")
            self.is_running = True
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to start container: {e.stderr}")
            return False
    
    def stop(self):
        """Stop the vLLM service."""
        try:
            # Stop container by name
            subprocess.run(
                ["docker", "stop", self.config.container_name],
                capture_output=True,
                check=True
            )
            logger.info("✅ Container stopped successfully")
            
            # Remove container
            subprocess.run(
                ["docker", "rm", self.config.container_name],
                capture_output=True,
                check=False  # Don't fail if container doesn't exist
            )
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ Error stopping container: {e.stderr}")
        
        finally:
            self.container_id = None
            self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """Get deployment status and health information."""
        try:
            # Get container status
            result = subprocess.run(
                ["docker", "inspect", self.config.container_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            container_info = json.loads(result.stdout)[0]
            state = container_info["State"]
            
            status = {
                "container_running": state["Running"],
                "exit_code": state.get("ExitCode", 0),
                "started_at": state.get("StartedAt"),
                "health_status": state.get("Health", {}).get("Status", "unknown"),
                "restart_count": container_info["RestartCount"]
            }
            
            # Get service health if container is running
            if state["Running"]:
                try:
                    health_response = requests.get(
                        f"http://localhost:{self.config.port}/health",
                        timeout=5
                    )
                    status["service_healthy"] = health_response.status_code == 200
                    status["service_response_time"] = health_response.elapsed.total_seconds()
                    
                except requests.RequestException:
                    status["service_healthy"] = False
                    status["service_response_time"] = None
            
            return status
            
        except subprocess.CalledProcessError:
            return {"container_running": False, "error": "Container not found"}
    
    def get_logs(self, lines: int = 50) -> str:
        """Get container logs."""
        try:
            result = subprocess.run(
                ["docker", "logs", "--tail", str(lines), self.config.container_name],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            return f"Error getting logs: {e.stderr}"
    
    def wait_for_healthy(self, timeout: int = 300) -> bool:
        """Wait for the service to become healthy."""
        logger.info(f"⏳ Waiting for service to become healthy (timeout: {timeout}s)...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self._shutdown_event.is_set():
                logger.info("Shutdown requested, stopping health check wait")
                return False
            
            status = self.get_status()
            
            if status.get("service_healthy", False):
                logger.info("✅ Service is healthy and ready!")
                return True
            
            if not status.get("container_running", False):
                logger.error("❌ Container stopped unexpectedly")
                logger.error(f"Container logs:\n{self.get_logs()}")
                return False
            
            time.sleep(5)
        
        logger.error(f"❌ Service failed to become healthy within {timeout}s")
        return False

class LoadBalancedDeployment:
    """Manage multiple vLLM instances with load balancing."""
    
    def __init__(self, base_config: DeploymentConfig, num_instances: int = 2):
        self.base_config = base_config
        self.num_instances = num_instances
        self.deployments: List[DockerDeploymentManager] = []
        self.nginx_container_id: Optional[str] = None
        
        # Create deployment instances
        for i in range(num_instances):
            config = DeploymentConfig(**asdict(base_config))
            config.port = base_config.port + i
            config.container_name = f"{base_config.container_name}-{i}"
            
            deployment = DockerDeploymentManager(config)
            self.deployments.append(deployment)
    
    def create_nginx_config(self) -> str:
        """Create NGINX configuration for load balancing."""
        upstream_servers = "\n".join([
            f"        server localhost:{self.base_config.port + i};"
            for i in range(self.num_instances)
        ])
        
        nginx_config = f"""
events {{
    worker_connections 1024;
}}

http {{
    upstream vllm_backend {{
{upstream_servers}
    }}
    
    server {{
        listen {self.base_config.port};
        
        location / {{
            proxy_pass http://vllm_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings for long-running requests
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }}
        
        location /health {{
            access_log off;
            return 200 "healthy\\n";
            add_header Content-Type text/plain;
        }}
    }}
}}
"""
        return nginx_config
    
    def deploy_with_load_balancing(self) -> bool:
        """Deploy multiple vLLM instances with NGINX load balancer."""
        logger.info(f"🚀 Deploying {self.num_instances} vLLM instances with load balancing...")
        
        # Deploy all vLLM instances
        successful_deployments = 0
        
        for i, deployment in enumerate(self.deployments):
            logger.info(f"Deploying instance {i + 1}/{self.num_instances}...")
            
            if deployment.deploy():
                if deployment.wait_for_healthy(timeout=180):
                    successful_deployments += 1
                    logger.info(f"✅ Instance {i + 1} is healthy")
                else:
                    logger.error(f"❌ Instance {i + 1} failed to become healthy")
            else:
                logger.error(f"❌ Failed to deploy instance {i + 1}")
        
        if successful_deployments == 0:
            logger.error("❌ No instances deployed successfully")
            return False
        
        logger.info(f"✅ {successful_deployments}/{self.num_instances} instances deployed successfully")
        
        # Deploy NGINX load balancer
        nginx_config = self.create_nginx_config()
        
        # Write NGINX config
        os.makedirs("nginx", exist_ok=True)
        with open("nginx/nginx.conf", "w") as f:
            f.write(nginx_config)
        
        # Start NGINX container
        nginx_cmd = [
            "docker", "run", "-d",
            "--name", "vllm-nginx-lb",
            "-p", f"{self.base_config.port}:{self.base_config.port}",
            "-v", f"{os.getcwd()}/nginx/nginx.conf:/etc/nginx/nginx.conf:ro",
            "--restart", "unless-stopped",
            "nginx:alpine"
        ]
        
        try:
            result = subprocess.run(nginx_cmd, capture_output=True, text=True, check=True)
            self.nginx_container_id = result.stdout.strip()
            logger.info(f"✅ NGINX load balancer started: {self.nginx_container_id[:12]}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to start NGINX load balancer: {e.stderr}")
            return False
    
    def stop_all(self):
        """Stop all instances and load balancer."""
        logger.info("🛑 Stopping load balanced deployment...")
        
        # Stop NGINX
        if self.nginx_container_id:
            try:
                subprocess.run(["docker", "stop", "vllm-nginx-lb"], check=True)
                subprocess.run(["docker", "rm", "vllm-nginx-lb"], check=False)
                logger.info("✅ NGINX load balancer stopped")
            except subprocess.CalledProcessError as e:
                logger.error(f"Error stopping NGINX: {e}")
        
        # Stop all vLLM instances
        for i, deployment in enumerate(self.deployments):
            try:
                deployment.stop()
                logger.info(f"✅ Instance {i + 1} stopped")
            except Exception as e:
                logger.error(f"Error stopping instance {i + 1}: {e}")

class MonitoringService:
    """Service for monitoring vLLM deployment metrics."""
    
    def __init__(self, deployment: DockerDeploymentManager):
        self.deployment = deployment
        self.metrics_history: List[Dict] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.should_monitor = False
    
    def start_monitoring(self, interval: int = 30):
        """Start monitoring service metrics."""
        self.should_monitor = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"📊 Monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring service."""
        self.should_monitor = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("📊 Monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.should_monitor:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 100 measurements
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Log important metrics
                if metrics.get("service_healthy", False):
                    response_time = metrics.get("service_response_time", 0)
                    logger.info(f"📊 Health check: OK ({response_time*1000:.1f}ms)")
                else:
                    logger.warning("📊 Health check: FAILED")
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
            
            time.sleep(interval)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect current metrics."""
        timestamp = time.time()
        status = self.deployment.get_status()
        
        metrics = {
            "timestamp": timestamp,
            **status
        }
        
        # Try to get vLLM-specific metrics if available
        if status.get("service_healthy", False):
            try:
                metrics_response = requests.get(
                    f"http://localhost:{self.deployment.config.port}/metrics",
                    timeout=5
                )
                if metrics_response.status_code == 200:
                    metrics["vllm_metrics_available"] = True
                    # Parse Prometheus metrics (simplified)
                    metrics["vllm_metrics_raw"] = metrics_response.text
            except requests.RequestException:
                metrics["vllm_metrics_available"] = False
        
        return metrics
    
    def get_summary_report(self) -> str:
        """Generate a summary report of monitoring data."""
        if not self.metrics_history:
            return "No monitoring data available"
        
        recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
        
        healthy_count = sum(1 for m in recent_metrics if m.get("service_healthy", False))
        health_percentage = (healthy_count / len(recent_metrics)) * 100
        
        response_times = [
            m["service_response_time"] for m in recent_metrics 
            if m.get("service_response_time") is not None
        ]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        report = f"""
📊 MONITORING SUMMARY REPORT
{'=' * 30}
Time Range: Last {len(recent_metrics)} measurements
Health Status: {health_percentage:.1f}% healthy
Average Response Time: {avg_response_time*1000:.1f}ms
Total Measurements: {len(self.metrics_history)}
        """
        
        return report.strip()

def demonstrate_single_instance_deployment():
    """Demonstrate single instance production deployment."""
    print("🚀 SINGLE INSTANCE DEPLOYMENT DEMO")
    print("=" * 40)
    
    # Create configuration
    config = DeploymentConfig(
        model_name="cardiffnlp/twitter-roberta-base-hate-latest",
        port=8000,
        max_model_len=2048,
        gpu_memory_utilization=0.8,  # Conservative for demo
        health_check_interval=30
    )
    
    print(f"📋 Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Port: {config.port}")
    print(f"   Max Length: {config.max_model_len}")
    print(f"   GPU Memory: {config.gpu_memory_utilization * 100}%")
    
    # Create deployment manager
    deployment = DockerDeploymentManager(config)
    
    # Check Docker availability
    if not deployment.is_docker_available():
        print("❌ Docker not available. Please install Docker first.")
        return
    
    try:
        # Deploy service
        print(f"\n🚀 Deploying vLLM service...")
        if deployment.deploy():
            print("✅ Deployment initiated successfully")
            
            # Wait for health
            if deployment.wait_for_healthy(timeout=300):
                print("✅ Service is healthy and ready!")
                
                # Start monitoring
                monitor = MonitoringService(deployment)
                monitor.start_monitoring(interval=30)
                
                # Demonstrate some requests
                demonstrate_production_requests(config.port)
                
                # Show monitoring report
                time.sleep(10)  # Let monitoring collect some data
                print(monitor.get_summary_report())
                
                # Stop monitoring
                monitor.stop_monitoring()
                
            else:
                print("❌ Service failed to become healthy")
                print(f"Container logs:\n{deployment.get_logs()}")
        
        else:
            print("❌ Deployment failed")
    
    except KeyboardInterrupt:
        print("\n⏹️ Deployment interrupted by user")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        deployment.stop()
        print("✅ Cleanup completed")

def demonstrate_load_balanced_deployment():
    """Demonstrate load balanced production deployment."""
    print("\n🔄 LOAD BALANCED DEPLOYMENT DEMO")
    print("=" * 40)
    
    # Create base configuration
    base_config = DeploymentConfig(
        model_name="cardiffnlp/twitter-roberta-base-hate-latest",
        port=8000,
        max_model_len=1024,  # Smaller for multiple instances
        gpu_memory_utilization=0.4,  # Split GPU memory
        health_check_interval=30
    )
    
    # Create load balanced deployment
    lb_deployment = LoadBalancedDeployment(base_config, num_instances=2)
    
    try:
        print(f"🚀 Deploying 2 vLLM instances with load balancing...")
        
        if lb_deployment.deploy_with_load_balancing():
            print("✅ Load balanced deployment completed!")
            
            # Test load balancing
            time.sleep(5)
            demonstrate_production_requests(base_config.port, num_requests=10)
        
        else:
            print("❌ Load balanced deployment failed")
    
    except KeyboardInterrupt:
        print("\n⏹️ Load balanced deployment interrupted")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up load balanced deployment...")
        lb_deployment.stop_all()
        print("✅ Load balanced cleanup completed")

def demonstrate_production_requests(port: int, num_requests: int = 5):
    """Demonstrate production API requests."""
    print(f"\n💬 Testing production API ({num_requests} requests)...")
    
    base_url = f"http://localhost:{port}/v1"
    
    test_texts = [
        "I love learning about artificial intelligence!",
        "This community is very supportive and welcoming.",
        "Great work on the project, very impressive results.",
        "Technology is advancing so quickly these days.",
        "Thank you for the helpful explanation and examples."
    ]
    
    for i, text in enumerate(test_texts[:num_requests], 1):
        try:
            payload = {
                "model": "hate-speech-detector",
                "prompt": f"Classify this text sentiment (positive/negative/neutral): {text}\nClassification:",
                "max_tokens": 5,
                "temperature": 0.1
            }
            
            start_time = time.time()
            response = requests.post(f"{base_url}/completions", json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                classification = result["choices"][0]["text"].strip()
                response_time = (end_time - start_time) * 1000
                
                print(f"✅ Request {i}: '{text[:40]}...' → {classification} ({response_time:.0f}ms)")
            else:
                print(f"❌ Request {i} failed: {response.status_code}")
        
        except requests.RequestException as e:
            print(f"❌ Request {i} error: {e}")
        
        time.sleep(1)  # Rate limiting

def save_deployment_configuration(config: DeploymentConfig, filename: str = "vllm_config.yaml"):
    """Save deployment configuration to YAML file."""
    config_dict = asdict(config)
    
    with open(filename, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)
    
    print(f"💾 Configuration saved to {filename}")

def load_deployment_configuration(filename: str = "vllm_config.yaml") -> DeploymentConfig:
    """Load deployment configuration from YAML file."""
    try:
        with open(filename, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return DeploymentConfig(**config_dict)
    
    except FileNotFoundError:
        print(f"⚠️ Configuration file {filename} not found, using defaults")
        return DeploymentConfig()

@contextmanager
def production_deployment_context(config: DeploymentConfig):
    """Context manager for production deployment lifecycle."""
    deployment = DockerDeploymentManager(config)
    monitor = None
    
    try:
        # Deploy
        if deployment.deploy() and deployment.wait_for_healthy():
            # Start monitoring
            monitor = MonitoringService(deployment)
            monitor.start_monitoring()
            
            yield deployment, monitor
        else:
            raise RuntimeError("Failed to deploy service")
    
    finally:
        # Cleanup
        if monitor:
            monitor.stop_monitoring()
        deployment.stop()

if __name__ == "__main__":
    """Main execution with production deployment examples."""
    
    print("🏭 vLLM Production Deployment Examples")
    print("🎯 Focus: Offline Docker Deployment with Monitoring")
    print("=" * 55)
    
    try:
        # Single instance deployment
        demonstrate_single_instance_deployment()
        
        # Prompt for load balanced demo
        if input("\n⏭️ Run load balanced deployment demo? (y/N): ").lower().startswith('y'):
            demonstrate_load_balanced_deployment()
        
        # Save example configuration
        config = DeploymentConfig()
        save_deployment_configuration(config, "example_vllm_config.yaml")
        
        print("\n✅ Production deployment examples completed!")
        print("💡 Key takeaways:")
        print("   • Docker enables consistent offline deployment")
        print("   • Health checks ensure service reliability")
        print("   • Load balancing improves throughput and availability")
        print("   • Monitoring provides operational insights")
        print("   • Configuration management simplifies deployment")
        
    except KeyboardInterrupt:
        print("\n👋 Deployment examples interrupted. Goodbye!")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print("💡 Check logs for detailed error information")
        sys.exit(1)