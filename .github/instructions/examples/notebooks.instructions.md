# Jupyter Notebook Instructions for HF Transformer Trove Examples

## Scope
This instruction file applies specifically to all Jupyter notebook files (`.ipynb`) in the `examples/` directory.

## Repository Focus Areas
All notebook examples should align with the repository's focus areas:
- **Implementation with HF (Hugging Face)**: Primary framework and ecosystem for all implementations
- **NLP (Natural Language Processing)**: Core domain focus with comprehensive coverage of NLP tasks
- **Hate Speech Detection (Preferred)**: Emphasized application area for practical examples and use cases

## Educational Jupyter Notebook Guidelines

### Notebook Structure Requirements
- **Always start with badges**: Include "Open in Colab", "Open with SageMaker", and "View on GitHub" badges at the top
  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vuhung16au/hf-transformer-trove/blob/main/examples/NOTEBOOK_NAME.ipynb)
  [![Open with SageMaker](https://img.shields.io/badge/Open%20with-SageMaker-orange?logo=amazonaws)](https://studiolab.sagemaker.aws/import/github/vuhung16au/hf-transformer-trove/blob/main/examples/NOTEBOOK_NAME.ipynb)
  [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-blue?logo=github)](https://github.com/vuhung16au/hf-transformer-trove/blob/main/examples/NOTEBOOK_NAME.ipynb)
  ```

- **Title and Learning Objectives**: Start each notebook with a clear title and list of learning objectives
- **Progressive Structure**: Follow the consistent cell structure: Import → Load Data → Process → Model → Evaluate → Visualize
- **Section Summaries**: End each major section with summary points or key takeaways
- **Always end with author footer**: Include author profile, LinkedIn, and GitHub links at the end
  ```markdown
  ---
  
  ## About the Author
  
  **Vu Hung Nguyen** - AI Engineer & Researcher
  
  Connect with me:
  - 🌐 **Website**: [vuhung16au.github.io](https://vuhung16au.github.io/)
  - 💼 **LinkedIn**: [linkedin.com/in/nguyenvuhung](https://www.linkedin.com/in/nguyenvuhung/)
  - 💻 **GitHub**: [github.com/vuhung16au](https://github.com/vuhung16au/)
  
  *This notebook is part of the [HF Transformer Trove](https://github.com/vuhung16au/hf-transformer-trove) educational series.*
  ```

### Educational Content Standards
- **Explain Before Code**: Use markdown cells to explain concepts before implementing them in code cells
- **Educational Comments**: Include comprehensive comments that explain ML/NLP concepts, not just code functionality
- **Visual Learning**: Include visualizations, diagrams, and plots wherever possible to aid understanding
- **Mathematical Context**: Use LaTeX equations in markdown cells for mathematical explanations
- **Architecture Diagrams**: Use Mermaid diagrams for visual explanations of model architectures or workflows

### Code Quality for Learning
- **Comprehensive Error Handling**: Handle errors gracefully with informative messages that help learning
- **Device Awareness**: Automatically detect and use optimal device (CUDA, MPS, CPU)
- **Memory Efficiency**: Include memory considerations and optimization tips
- **Performance Timing**: Add timing information for expensive operations to teach performance awareness

### Hugging Face Best Practices
- **Modern APIs**: Use the latest Hugging Face APIs and patterns
- **Auto Classes**: Prefer `AutoModel`, `AutoTokenizer`, `AutoConfig` for flexibility
- **Pipeline Usage**: Demonstrate both high-level pipelines and low-level implementations
- **Model Loading**: Show proper model and tokenizer loading patterns
- **Dataset Integration**: Use Hugging Face Datasets library when appropriate

### Preferred Models and Datasets for Notebook Examples

#### Hate Speech Detection Models (use in order of preference):
1. **cardiffnlp/twitter-roberta-base-hate-latest** - Best for social media content examples
2. **facebook/roberta-hate-speech-dynabench-r4-target** - For robust hate speech classification
3. **GroNLP/hateBERT** - Educational examples of specialized architectures
4. **Hate-speech-CNERG/dehatebert-mono-english** - DeBERTa-based hate speech detection
5. **cardiffnlp/twitter-roberta-base-offensive** - Alternative for offensive language detection

#### Hate Speech Datasets (use in order of preference):
1. **tdavidson/hate_speech_offensive** - Standard benchmark for classification examples
2. **Hate-speech-CNERG/hatexplain** - Includes explanations for interpretability notebooks
3. **TrustAIRLab/HateBenchSet** - Comprehensive evaluation examples
4. **iamollas/ethos** - For bias analysis and ethical AI notebooks

#### Standard Notebook Cell Pattern:
```python
# 📱 Load preferred hate speech detection model with TPU-aware device selection
model_name = "cardiffnlp/twitter-roberta-base-hate-latest"
print(f"🔄 Loading {model_name}...")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Get optimal device (includes TPU detection for Colab)
device = get_device()

# Create hate speech detection pipeline with device-aware setup
hate_speech_classifier = pipeline(
    "text-classification",
    model=model_name,
    tokenizer=model_name,
    device=device if device.type != 'xla' else 0  # Handle TPU device mapping
)

print("✅ Hate speech detection model loaded successfully!")
```

### Platform Compatibility
- **Multi-platform**: Ensure notebooks run on local environments, Google Colab, AWS SageMaker Studio, and Kaggle
- **Google Colab TPU**: Always prefer TPU when available in Colab for training and inference
- **SageMaker Studio Requirements**: Notebooks should be compatible with AWS SageMaker Studio Lab (https://studiolab.sagemaker.aws/)
- **Dependency Management**: Include clear installation instructions for any additional packages
- **Credential Handling**: Use secure patterns for API keys and credentials (environment variables, not hardcoded)
- **Resource Awareness**: Consider computational requirements and provide alternatives for resource-constrained environments

### Google Colab TPU-Specific Guidelines
- **Runtime Selection**: Include instructions for users to select TPU runtime in Colab
- **TPU Libraries**: Handle `torch_xla` import failures gracefully with educational messages
- **Batch Size Optimization**: Recommend larger batch sizes for TPU to maximize utilization
- **Educational Context**: Explain why TPU is preferred in Colab environments

### Learning Progressive Complexity
- **Basic to Advanced**: Structure content from basic concepts to advanced implementations
- **Multiple Approaches**: Show different ways to solve the same problem when educational
- **Before/After Comparisons**: Demonstrate improvements and optimizations
- **Real-world Context**: Connect examples to practical applications

### Documentation Standards
- **Cell Documentation**: Each code cell should have a brief explanation of its purpose
- **Variable Naming**: Use descriptive variable names that explain ML concepts
- **Output Interpretation**: Explain what outputs mean in the context of the learning objective
- **Next Steps**: End notebooks with suggestions for further learning or experimentation

### Avoid in Notebooks
- Overly complex examples that obscure learning goals
- Deprecated APIs or outdated patterns
- Hard-coded paths or credentials
- Examples requiring excessive computational resources without alternatives
- Code without educational context or explanation
- Missing error handling or edge case considerations

### Integration with Repository
- **Cross-references**: Link to related documentation in `docs/` directory
- **Learning Path**: Reference previous and next notebooks in the learning sequence
- **External Resources**: Link to official Hugging Face documentation and tutorials
- **Community Resources**: Reference PyTorch Mastery and NLP Learning Journey repositories when appropriate

## Quality Assurance
- Test notebooks on multiple platforms before finalizing (local, Colab, SageMaker Studio)
- Verify all imports and dependencies are available across platforms
- Ensure educational clarity without sacrificing technical accuracy
- Validate that examples work with current library versions
- Test SageMaker Studio compatibility via import link functionality