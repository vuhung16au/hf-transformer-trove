# Code Style Instructions for HF Transformer Trove

## Scope
This instruction file applies to all Python code files (`.py`) and code cells in Jupyter notebooks throughout the repository.

## Repository Focus Areas
All code implementations should align with the repository's focus areas:
- **Implementation with HF (Hugging Face)**: Primary framework and ecosystem for all implementations
- **NLP (Natural Language Processing)**: Core domain focus with comprehensive coverage of NLP tasks
- **Hate Speech Detection (Preferred)**: Emphasized application area for practical examples and use cases

## Python Code Style Guidelines

### General Python Standards
- **Python Version**: Use Python 3.8+ features
- **PEP 8 Compliance**: Follow PEP 8 style guidelines
- **Object-Oriented Approach**: Prefer OOP implementation - use classes and objects when appropriate
- **Type Hints**: Include type hints where beneficial for learning and code clarity
- **Meaningful Names**: Use variable names that explain ML concepts and domain terminology
- **Simplicity**: Keep source code as simple as possible while being educational

### Documentation Standards
- **Docstrings**: Add comprehensive docstrings for functions explaining ML terminology and concepts
- **Comments**: Include educational comments that explain the "why" behind code decisions
- **Code Examples**: All code should be self-contained and runnable
- **Educational Context**: Every code block should have clear educational purpose

### Error Handling Requirements
- **Comprehensive Error Handling**: Handle errors gracefully with informative messages
- **Educational Error Messages**: Error messages should help users learn and understand common pitfalls
- **Exception Patterns**: Show proper exception handling patterns in all examples
- **Troubleshooting**: Include troubleshooting tips in comments for common issues

### Code Structure Patterns
```python
# Example of educational code structure
from typing import Optional, Union, List
import torch
from transformers import AutoTokenizer, AutoModel

class EducationalModelWrapper:
    """
    Educational wrapper for Hugging Face models.
    
    This class demonstrates best practices for model loading,
    device management, and inference patterns.
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize the model wrapper with educational error handling."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = self._get_device(device)
            self.model.to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to load model {model_name}: {e}")
    
    def _get_device(self, device: Optional[str] = None) -> torch.device:
        """Get the best available device with educational commentary."""
        if device:
            return torch.device(device)
        
        # Educational device selection with priority explanation
        if torch.cuda.is_available():
            print("🚀 Using CUDA GPU for optimal performance")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("🍎 Using Apple MPS for Apple Silicon optimization")
            return torch.device("mps")
        else:
            print("💻 Using CPU - consider GPU for better performance")
            return torch.device("cpu")
```

### Learning-Focused Code Features
- **Progressive Complexity**: Structure code from basic concepts to advanced implementations
- **Before/After Comparisons**: Show improvements and optimizations clearly
- **Multiple Approaches**: Demonstrate different ways to solve the same problem when educational
- **Performance Awareness**: Include timing information and memory considerations
- **Real-world Context**: Connect code examples to practical applications

### Educational Comment Style
```python
# Load pre-trained BERT model for sequence classification
# This downloads ~400MB model if not cached locally
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,  # Binary classification (positive/negative)
    output_attentions=False,  # Save memory by not returning attention weights
    output_hidden_states=False,  # Save memory by not returning hidden states
)

# Automatically detect and use the best available device
# Priority: CUDA > MPS (Apple Silicon) > CPU
device = get_device()
model = model.to(device)
print(f"📱 Using device: {device}")

# Enable training mode for fine-tuning
model.train()  # Sets model to training mode (enables dropout, batch norm updates)
```

### Code Quality Standards
- **Modularity**: Break code into logical, reusable functions and classes
- **Testing Patterns**: Include validation and testing examples where appropriate
- **Resource Management**: Demonstrate proper memory and computational efficiency
- **Version Compatibility**: Ensure compatibility with specified library versions

### Avoid in Code
- Overly complex examples that obscure learning objectives
- Hard-coded paths, credentials, or configuration values
- Code without educational context or explanation
- Missing error handling for common failure scenarios
- Deprecated APIs or outdated patterns
- Examples requiring excessive computational resources without alternatives