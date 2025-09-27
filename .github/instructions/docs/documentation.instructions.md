# Documentation Instructions for HF Transformer Trove

## Scope
This instruction file applies specifically to all Markdown documentation files (`.md`) in the `docs/` directory.

## Repository Focus Areas
All documentation should align with the repository's focus areas:
- **Implementation with HF (Hugging Face)**: Primary framework and ecosystem for all implementations
- **NLP (Natural Language Processing)**: Core domain focus with comprehensive coverage of NLP tasks
- **Hate Speech Detection (Preferred)**: Emphasized application area for practical examples and use cases

## Documentation Standards

### Content Structure
- **Clear Hierarchy**: Use consistent header levels (H1 for title, H2 for main sections, H3 for subsections)
- **Table of Contents**: Include TOC for longer documents (>5 sections)
- **Introduction**: Start with brief overview explaining the document's purpose and scope
- **Learning Objectives**: Clearly state what readers will learn from the document

### Educational Writing Style
- **Accessible Language**: Explain technical concepts clearly, assuming readers have ML/PyTorch/NLP basics
- **Progressive Complexity**: Structure content from basic concepts to advanced topics
- **Contextual Definitions**: Define technical terms in context rather than assuming prior knowledge
- **Practical Examples**: Include concrete code examples that readers can understand and apply

### Technical Content Requirements
- **Code Examples**: All code samples must be:
  - Self-contained and runnable
  - Well-commented with educational explanations
  - Use modern Hugging Face APIs and best practices
  - Include proper error handling
  - Show device awareness (CUDA/MPS/CPU optimization)

- **Code Blocks**: Use proper syntax highlighting:
  ```python
  # Educational code with comprehensive comments
  from transformers import AutoTokenizer, AutoModel
  
  # Load pre-trained model - explain why this specific model
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased")
  ```

### Hugging Face Ecosystem Focus
- **Library Integration**: Show how different HF libraries work together (transformers, datasets, tokenizers)
- **Model Hub References**: Link to relevant models on Hugging Face Hub with explanations of when to use them
- **API Documentation**: Reference official HF documentation for deeper exploration
- **Best Practices**: Include HF community best practices and common patterns

### Visual and Interactive Elements
- **Diagrams**: Use Mermaid diagrams for:
  - Model architectures
  - Data flow processes
  - Training pipelines
  - Workflow visualizations

- **Mathematical Notation**: Use LaTeX for mathematical explanations:
  ```latex
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  ```

- **Tables**: Use tables for:
  - Parameter comparisons
  - Model specifications
  - Performance metrics
  - API reference summaries

### Cross-Reference and Navigation
- **Internal Links**: Link to related sections within the same document
- **Repository Links**: Reference relevant notebook examples in `examples/` directory
- **External References**: Link to:
  - Official Hugging Face documentation
  - Research papers for advanced topics
  - PyTorch Mastery documentation for basic concepts
  - NLP Learning Journey documentation for NLP fundamentals

### Learning Enhancement Features
- **Key Takeaways**: Include summary boxes for important concepts:
  > **Key Takeaway**: Brief explanation of the most important concept

- **Pro Tips**: Add practical advice boxes:
  > 💡 **Pro Tip**: Practical advice for real-world applications

- **Common Pitfalls**: Warn about frequent mistakes:
  > ⚠️ **Common Pitfall**: Explanation of what to avoid and why

- **Performance Notes**: Include performance considerations:
  > 🚀 **Performance**: Tips for optimization and efficiency

### Code Quality Standards
- **Type Hints**: Include type hints in code examples for educational clarity
- **Error Handling**: Show proper exception handling patterns
- **Resource Management**: Demonstrate memory and computational efficiency
- **Testing Patterns**: Include examples of validation and testing approaches

### Accessibility and Compatibility
- **Inclusive Language**: Use clear, inclusive language accessible to diverse audiences
- **Multiple Learning Styles**: Combine text, code, visuals, and examples
- **Scalability**: Provide examples that work on different computational resources
- **Version Awareness**: Specify library versions when relevant for compatibility

### Documentation Maintenance
- **Accuracy**: Ensure all code examples are tested and functional
- **Currency**: Keep content updated with latest HF library versions
- **Consistency**: Maintain consistent terminology and style across all documentation
- **Completeness**: Cover topics thoroughly without overwhelming detail

### Integration Guidelines
- **Notebook Alignment**: Ensure documentation complements practical notebook examples
- **Learning Path**: Structure documents to support the repository's learning progression
- **Prerequisite Clarity**: Clearly state required background knowledge
- **Next Steps**: Guide readers to related materials and advanced topics

### Avoid in Documentation
- Overly technical jargon without explanation
- Deprecated APIs or outdated examples
- Incomplete code samples that won't run
- Missing context for why techniques are useful
- Inconsistent formatting and style
- Examples that require unavailable resources or credentials