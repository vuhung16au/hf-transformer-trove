# Jupyter Notebook Instructions for HF Transformer Trove Examples

## Scope
This instruction file applies specifically to all Jupyter notebook files (`.ipynb`) in the `examples/` directory.

## Educational Jupyter Notebook Guidelines

### Notebook Structure Requirements
- **Always start with badges**: Include "Open in Colab" and "View on GitHub" badges at the top
  ```markdown
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vuhung16au/hf-transformer-trove/blob/main/examples/NOTEBOOK_NAME.ipynb)
  [![View on GitHub](https://img.shields.io/badge/View_on-GitHub-blue?logo=github)](https://github.com/vuhung16au/hf-transformer-trove/blob/main/examples/NOTEBOOK_NAME.ipynb)
  ```

- **Title and Learning Objectives**: Start each notebook with a clear title and list of learning objectives
- **Progressive Structure**: Follow the consistent cell structure: Import → Load Data → Process → Model → Evaluate → Visualize
- **Section Summaries**: End each major section with summary points or key takeaways

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

### Platform Compatibility
- **Multi-platform**: Ensure notebooks run on local environments, Google Colab, and Kaggle
- **Dependency Management**: Include clear installation instructions for any additional packages
- **Credential Handling**: Use secure patterns for API keys and credentials (environment variables, not hardcoded)
- **Resource Awareness**: Consider computational requirements and provide alternatives for resource-constrained environments

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
- Test notebooks on multiple platforms before finalizing
- Verify all imports and dependencies are available
- Ensure educational clarity without sacrificing technical accuracy
- Validate that examples work with current library versions