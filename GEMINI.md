# GEMINI AI Assistant Guidelines

## Overview
This document provides specific guidelines for AI assistants (particularly Google's Gemini) when working with the HF Transformer Trove repository. These guidelines ensure consistent, educational, and high-quality assistance for learning the Hugging Face ecosystem.

## Repository Context
**HF Transformer Trove** is an educational resource focused on learning the Hugging Face ecosystem and natural language processing. The repository serves as comprehensive study notes with practical implementations for ML practitioners and students.

### Target Audience
- ML practitioners familiar with PyTorch and NLP basics
- Students learning the Hugging Face ecosystem
- Developers seeking practical, educational examples
- Researchers looking for implementation patterns

## Core Principles

### 1. Educational First
- **Always prioritize learning value** over code brevity
- Explain concepts thoroughly with context and rationale
- Use examples that demonstrate both the "how" and the "why"
- Connect implementations to underlying ML/NLP theory when relevant

### 2. Practical and Actionable
- Provide complete, runnable code examples
- Include error handling and edge case considerations
- Show device awareness (CUDA/MPS/CPU optimization)
- Demonstrate real-world applications and use cases

### 3. Modern Best Practices
- Use latest Hugging Face APIs and patterns
- Prefer `Auto*` classes for flexibility and best practices
- Show both high-level (pipeline) and low-level implementations
- Include performance and memory optimization tips

## Content Guidelines

### Code Generation
When generating code:
```python
# ✅ GOOD: Educational with comprehensive comments
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained BERT model for sequence classification
# This downloads the model if not cached locally (~400MB)
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,        # Binary classification
    output_attentions=False,    # Save memory by not returning attention weights
    output_hidden_states=False  # Save memory by not returning hidden states
)

# Automatically detect and use the best available device
device = torch.device("cuda" if torch.cuda.is_available() 
                     else "mps" if torch.backends.mps.is_available() 
                     else "cpu")
model = model.to(device)
print(f"Using device: {device}")
```

### Documentation Writing
When creating documentation:
- Start with clear learning objectives
- Use progressive complexity (basic → advanced)
- Include visual elements (diagrams, tables, examples)
- Cross-reference related materials in the repository
- End with practical next steps and further resources

### Notebook Structure
When working with Jupyter notebooks:
1. **Header**: Include Colab and GitHub badges
2. **Introduction**: Title, learning objectives, prerequisites
3. **Imports**: Organized, well-commented imports
4. **Content Sections**: Concept explanation followed by implementation
5. **Visualizations**: Charts, plots, and diagrams for understanding
6. **Summary**: Key takeaways and next steps

## Technical Specifications

### Hugging Face Ecosystem
- **Transformers**: Focus on `transformers` library as primary
- **Datasets**: Use `datasets` library for data handling
- **Tokenizers**: Demonstrate different tokenization approaches
- **Model Hub**: Reference and use models from Hugging Face Hub
- **Pipelines**: Show both pipeline and manual implementations

### Code Quality Standards
- **Type Hints**: Include type annotations for educational clarity
- **Error Handling**: Comprehensive exception handling with helpful messages
- **Resource Management**: Memory-efficient patterns and cleanup
- **Testing**: Include validation and testing examples
- **Documentation**: Docstrings explaining ML concepts and parameters

### Device and Performance Awareness
```python
# Example of device-aware implementation
def get_device():
    """Get the best available device for PyTorch operations."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Always include timing for educational purposes
import time
start_time = time.time()
# ... model operations ...
print(f"Operation completed in {time.time() - start_time:.2f} seconds")
```

## Response Patterns

### When Explaining Concepts
1. **Context First**: Explain why the concept is important
2. **Simple Definition**: Clear, accessible explanation
3. **Code Example**: Practical implementation
4. **Common Pitfalls**: What to avoid and why
5. **Further Reading**: Links to official documentation

### When Providing Code
1. **Comment Thoroughly**: Explain each significant step
2. **Show Alternatives**: When multiple approaches exist
3. **Include Validation**: Show how to verify results
4. **Performance Notes**: Memory and speed considerations
5. **Error Handling**: Graceful failure with helpful messages

### When Debugging Issues
1. **Understand Context**: Ask about environment, versions, data
2. **Systematic Approach**: Check common issues first
3. **Educational Debugging**: Explain why errors occur
4. **Prevention**: How to avoid similar issues
5. **Best Practices**: Improved patterns for the future

## Integration with Repository Structure

### Examples Directory (`examples/`)
- Reference existing notebooks when relevant
- Maintain consistency with established patterns
- Build upon previous notebook concepts progressively
- Include cross-references to related examples

### Documentation Directory (`docs/`)
- Link to comprehensive explanations for complex topics
- Reference key-terms.md and best-practices.md appropriately
- Maintain consistency with documented standards
- Contribute to the knowledge base structure

### Learning Path Alignment
Follow the repository's learning progression:
1. **Basic HF Introduction** (01_intro_hf_transformers.ipynb)
2. **Tokenization** (02_tokenizers.ipynb)  
3. **Datasets** (03_datasets_library.ipynb)
4. **Integration Projects** (04_mini_project.ipynb)
5. **Fine-tuning** (05_fine_tuning_trainer.ipynb)
6. **Advanced Applications** (07_summarization.ipynb, etc.)

## Quality Assurance

### Before Providing Assistance
- ✅ Verify code examples are complete and runnable
- ✅ Check that explanations are educational and accurate
- ✅ Ensure compatibility with multiple platforms (local, Colab, Kaggle)
- ✅ Validate that suggestions follow HF best practices
- ✅ Confirm alignment with repository's educational goals

### Avoid
- ❌ Deprecated APIs or outdated patterns
- ❌ Hard-coded credentials or paths
- ❌ Overly complex examples that obscure learning
- ❌ Code without educational context
- ❌ Ignoring resource constraints and platform differences
- ❌ Breaking existing repository patterns and conventions

## Communication Style
- **Encouraging**: Support learning and experimentation
- **Clear**: Use accessible language while maintaining technical accuracy  
- **Comprehensive**: Provide complete information without overwhelming
- **Practical**: Focus on actionable advice and implementations
- **Educational**: Always explain the reasoning behind suggestions

Remember: The goal is to help users master the Hugging Face ecosystem through practical, well-explained examples that they can build upon for their own projects.