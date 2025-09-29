# Technical Preferences Instructions for HF Transformer Trove

## Scope
This instruction file defines the technical preferences and standards for all code implementations in the repository.

## Repository Focus Areas
All technical implementations should align with the repository's focus areas:
- **Implementation with HF (Hugging Face)**: Primary framework and ecosystem for all implementations
- **NLP (Natural Language Processing)**: Core domain focus with comprehensive coverage of NLP tasks
- **Hate Speech Detection (Preferred)**: Emphasized application area for practical examples and use cases

## Framework and Technology Preferences

### Deep Learning Framework
- **Primary Framework**: PyTorch over TensorFlow for all deep learning implementations
- **Justification**: Better educational value, more intuitive for learning, stronger HF ecosystem integration
- **Visualization**: Use TensorBoard integration for comprehensive training visualization when possible

### Hugging Face Ecosystem Preferences
- **Transformers Library**: Prefer `transformers` library over direct PyTorch/TensorFlow implementations
- **Auto Classes**: Use `AutoTokenizer`, `AutoModel`, `AutoConfig` for flexibility and best practices
- **Datasets Library**: Use `datasets` library for all data handling and preprocessing
- **Tokenizers Library**: Utilize `tokenizers` for efficient tokenization when needed
- **Trainer API**: Implement `Trainer` API for training workflows as the primary approach
- **Pipeline Approach**: Show both high-level pipeline and manual approaches where applicable
- **Application Focus**: Prioritize hate speech detection examples when demonstrating classification tasks

### Preferred Models and Datasets for Hate Speech Detection

#### Recommended Hate Speech Detection Models (in order of preference):
1. **cardiffnlp/twitter-roberta-base-hate-latest** - State-of-the-art Twitter-trained hate speech detection
2. **facebook/roberta-hate-speech-dynabench-r4-target** - Facebook's robust hate speech classifier
3. **GroNLP/hateBERT** - Specialized BERT model for hate speech detection (HateBERT)
4. **Hate-speech-CNERG/dehatebert-mono-english** - DeBERTa-based hate speech detection
5. **cardiffnlp/twitter-roberta-base-offensive** - Offensive language detection variant

#### Fallback Models (when hate speech models unavailable):
- **unitary/toxic-bert** - General toxicity detection
- **martin-ha/toxic-comment-model** - Multi-label toxic comment classifier

#### Recommended Hate Speech Datasets (in order of preference):
1. **tdavidson/hate_speech_offensive** - Davidson et al. hate speech and offensive language dataset
2. **Hate-speech-CNERG/hatexplain** - HatEXplain dataset with explanations
3. **TrustAIRLab/HateBenchSet** - Comprehensive hate speech benchmark dataset
4. **iamollas/ethos** - ETHOS hate speech detection dataset

## Device Awareness Implementation

### Device Detection Pattern

#### General Device Priority Order
**Standard Priority**: GPU (CUDA) > TPU (Google Colab) > MPS (Apple Silicon) > CPU

#### Google Colab Specific Policy
**When training or inferencing on Google Colab, always prefer TPU when available**

```python
import torch

# For Google Colab TPU compatibility
try:
    from google.colab import userdata
    import torch_xla.core.xla_model as xm
    COLAB_AVAILABLE = True
    TPU_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    TPU_AVAILABLE = False

def get_device() -> torch.device:
    """
    Get the best available device for PyTorch operations.
    
    Device Priority:
    - General: CUDA GPU > TPU (Colab only) > MPS (Apple Silicon) > CPU
    - Google Colab: Always prefer TPU when available
    
    Returns:
        torch.device: The optimal device for current hardware
    """
    # Google Colab: Always prefer TPU when available
    if COLAB_AVAILABLE and TPU_AVAILABLE:
        try:
            # Try to initialize TPU
            device = xm.xla_device()
            print("🔥 Using Google Colab TPU for optimal performance")
            print("💡 TPU is preferred in Colab for training and inference")
            return device
        except Exception as e:
            print(f"⚠️ TPU initialization failed: {e}")
            print("Falling back to GPU/CPU detection")
    
    # Standard device detection for other environments
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍎 Using Apple MPS for Apple Silicon optimization")
    else:
        device = torch.device("cpu")
        print("💻 Using CPU - consider GPU/TPU for better performance")
    
    return device

# Usage pattern in all code
device = get_device()
model = model.to(device)
```

### Memory Optimization Patterns
```python
# Enable memory-efficient attention for large models
model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half precision for memory efficiency
    device_map="auto",  # Automatically distribute model across available GPUs
)

# Gradient checkpointing for memory efficiency during training
model.gradient_checkpointing_enable()

# Clear cache based on device type
if torch.cuda.is_available():
    torch.cuda.empty_cache()
elif TPU_AVAILABLE and COLAB_AVAILABLE:
    # TPU memory management in Colab
    import gc
    gc.collect()
    print("🔄 TPU memory cleared")
```

### TPU-Specific Optimization (Google Colab)
```python
# TPU optimization patterns for Google Colab
def configure_tpu_training():
    """Configure TPU settings for optimal training in Google Colab."""
    if COLAB_AVAILABLE and TPU_AVAILABLE:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.distributed.xla_multiprocessing as xmp
        
        print("🔥 Configuring TPU for training")
        print(f"📊 TPU cores available: {xm.xrt_world_size()}")
        
        # TPU-specific memory management
        xm.mark_step()  # Ensure operations are sent to TPU
        
        return True
    return False

# Example usage in training loops for TPU
def tpu_training_step(model, batch, device):
    """Training step optimized for TPU usage."""
    if device.type == 'xla':
        import torch_xla.core.xla_model as xm
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        # Required for TPU: mark step after backward pass
        xm.mark_step()
        
        return loss.item()
    else:
        # Standard GPU/CPU training
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        return loss.item()
```

## Credential Management Standards

### Environment-Based Credential Loading
```python
import os
from typing import Optional

# For Google Colab compatibility
try:
    from google.colab import userdata
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

def get_api_key(key_name: str, required: bool = True) -> Optional[str]:
    """
    Get API key from environment variables or Google Colab secrets.
    
    Args:
        key_name: Name of the environment variable/secret
        required: Whether the key is required (raises error if missing)
    
    Returns:
        API key string or None if not required and not found
        
    Raises:
        ValueError: If required key is not found
    """
    api_key = None
    
    # Try Google Colab secrets first (when available)
    if COLAB_AVAILABLE:
        try:
            api_key = userdata.get(key_name)
            print(f"✅ Loaded {key_name} from Google Colab secrets")
        except:
            pass
    
    # Fall back to local environment variable
    if not api_key:
        api_key = os.getenv(key_name)
        if api_key:
            print(f"✅ Loaded {key_name} from environment variable")
    
    # Handle missing required keys
    if required and not api_key:
        raise ValueError(
            f"❌ {key_name} not found. Please set it in:\n"
            f"  - Local: .env.local file or environment variable\n"
            f"  - Colab: Secrets manager (🔑 icon in sidebar)"
        )
    
    return api_key

# Usage patterns
HF_API_KEY = get_api_key("HUGGING_FACE_API_KEY", required=False)
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY", required=False)
```

### Local Development Platform Preferences
- **Primary Platform**: Mac OS X (Apple Silicon chip) - Preferred for local development
- **Secondary Platform**: Windows - Second choice for local development  
- **Third Platform**: Linux - Third choice for local development
- **Justification**: Apple Silicon provides optimal performance for ML workloads with MPS acceleration, followed by Windows for accessibility, then Linux for server environments

### Platform-Specific Patterns
- **Local Development**: Credentials loaded from `.env.local` file
- **Google Colab**: Credentials loaded from Colab secrets manager + **Always prefer TPU for training/inference**
- **Kaggle**: Credentials loaded from Kaggle secrets
- **Never**: Hard-code credentials in notebooks or source code

### Google Colab TPU Optimization Guidelines
```python
# Google Colab TPU setup and usage pattern
def setup_colab_tpu_environment():
    """
    Set up Google Colab environment with TPU optimization.
    
    Returns:
        tuple: (device, is_tpu_available)
    """
    try:
        # Import TPU libraries
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        
        # Initialize TPU
        device = xm.xla_device()
        print("🔥 Google Colab TPU initialized successfully")
        print("💡 TPU provides significant speedup for transformer training")
        print(f"📊 TPU cores: {xm.xrt_world_size()}")
        
        return device, True
        
    except ImportError:
        print("❌ TPU libraries not available")
        print("💡 Make sure to select TPU runtime in Colab: Runtime -> Change runtime type -> TPU")
        return get_device(), False
    except Exception as e:
        print(f"⚠️ TPU setup failed: {e}")
        print("🔄 Falling back to GPU/CPU")
        return get_device(), False

# Educational TPU usage pattern
def educational_tpu_example():
    """Educational example showing TPU usage in Google Colab."""
    print("📚 Google Colab TPU Usage Guidelines:")
    print("1. Always prefer TPU when available in Colab")
    print("2. TPU excels at large batch training")
    print("3. Use torch_xla.core.xla_model.mark_step() after loss.backward()")
    print("4. TPU works best with consistent tensor shapes")
    print("5. Monitor TPU utilization for optimal performance")
```

## Language and Internationalization Preferences

### Language Priority
1. **Primary Language**: English for all NLP tasks and documentation
2. **Secondary Language**: Vietnamese (when second language needed)
3. **Third Language**: Japanese (when third language option needed)

### Text Processing Standards
```python
# Example multilingual processing pattern
SUPPORTED_LANGUAGES = {
    'en': 'English (Primary)',
    'vi': 'Vietnamese (Secondary)', 
    'ja': 'Japanese (Third option)'
}

def process_multilingual_text(text: str, lang: str = 'en') -> str:
    """Process text with language-specific considerations."""
    if lang not in SUPPORTED_LANGUAGES:
        print(f"⚠️  Language {lang} not in preferred list: {list(SUPPORTED_LANGUAGES.keys())}")
        print(f"Defaulting to English (en)")
        lang = 'en'
    
    print(f"🌐 Processing text in {SUPPORTED_LANGUAGES[lang]}")
    return text
```

## Dependency Management Standards

### Python Package Manager Preferences
- **Primary Package Manager**: `uv` over `pip` for all Python package management
- **Justification**: uv provides faster installation, better dependency resolution, and improved caching
- **Fallback**: Use `pip` only when uv is not available or compatible

### Dependency Philosophy
- **Minimal Dependencies**: Keep dependencies focused and minimal
- **Version Pinning**: Pin versions for reproducibility in requirements.txt
- **Popular Packages**: Prefer stable, well-maintained packages
- **Educational Value**: Choose packages that enhance learning experience

### Standard Dependencies Pattern
```python
# Core ML dependencies (always include)
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Visualization dependencies (for educational notebooks)
import matplotlib.pyplot as plt
import seaborn as sns

# Utility dependencies (as needed)
from tqdm.auto import tqdm  # Progress bars
import time  # Performance timing
from typing import List, Dict, Optional, Union  # Type hints
```

### Dependency Installation Patterns
```python
# Check and install dependencies with educational feedback
def check_and_install_requirements():
    """Check and install required packages with educational feedback."""
    required_packages = [
        "torch",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.split('>=')[0])
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {missing_packages}")
        # In notebooks, show uv install command (preferred) with pip fallback
        print(f"Please run: uv add {' '.join(missing_packages)}")
        print(f"Or with pip: pip install {' '.join(missing_packages)}")
```

## Performance and Optimization Standards

### Timing and Profiling Patterns
```python
import time
from contextlib import contextmanager

@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations with educational output."""
    print(f"⏱️  Starting: {operation_name}")
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        print(f"✅ Completed: {operation_name} in {duration:.2f} seconds")

# Usage pattern
with timer("Model loading"):
    model = AutoModel.from_pretrained("bert-base-uncased")

with timer("Dataset processing"):
    processed_dataset = dataset.map(tokenize_function, batched=True)
```

### Memory Monitoring Patterns
```python
def print_gpu_memory_usage():
    """Print current GPU memory usage for educational purposes."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"🔋 GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
```

## Integration Requirements

### Cross-Repository References
- **PyTorch Mastery**: Link to [PyTorch Mastery Documentation](https://github.com/vuhung16au/pytorch-mastery/docs/) for basic concepts
- **NLP Learning Journey**: Reference [NLP Learning Journey Documentation](https://github.com/vuhung16au/nlp-learning-journey/docs/) for NLP fundamentals
- **Official HF Docs**: Always reference official Hugging Face documentation for deeper exploration

### Version Compatibility
- **Python**: 3.8+ required
- **PyTorch**: Latest stable version preferred
- **Transformers**: 4.20.0+ for modern APIs
- **Datasets**: 2.0.0+ for latest features