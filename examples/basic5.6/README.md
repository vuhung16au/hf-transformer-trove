# Basic 5.6: Semantic Search with FAISS

This directory contains educational materials for learning semantic search using FAISS and HuggingFace transformers.

## 📚 Contents

### Notebooks

1. **semantic-search.ipynb** - Comprehensive guide to semantic search
   - Introduction to semantic search concepts
   - Loading and preparing the GitHub issues dataset
   - Creating text embeddings with sentence transformers
   - Building FAISS indices for efficient similarity search
   - Performing semantic queries and analyzing results
   - Advanced FAISS index types and optimization
   - Production deployment best practices

## 🎯 Learning Objectives

After completing these materials, you will understand:

- How semantic search differs from traditional keyword-based search
- How to generate meaningful text embeddings using transformer models
- How to use FAISS for efficient similarity search at scale
- Different FAISS index types and when to use them
- Best practices for building production-ready semantic search systems

## 📋 Prerequisites

- Basic understanding of machine learning concepts
- Familiarity with Python and PyTorch
- Knowledge of NLP fundamentals
- Understanding of transformer architectures

## 🚀 Getting Started

### Option 1: Run in Google Colab (Recommended)
Click the "Open in Colab" badge at the top of the notebook to run it in Google Colab with free GPU/TPU access.

### Option 2: Run in AWS SageMaker Studio Lab
Click the "Open with SageMaker" badge to import and run in AWS SageMaker Studio Lab.

### Option 3: Run Locally

1. Install required packages:
```bash
pip install transformers datasets torch numpy pandas matplotlib seaborn faiss-cpu tqdm
```

2. For GPU support with FAISS:
```bash
pip install faiss-gpu
```

3. Open and run the notebook:
```bash
jupyter notebook semantic-search.ipynb
```

## 📦 Required Libraries

- **transformers** (>=4.35.0) - HuggingFace transformer models
- **datasets** (>=2.14.0) - HuggingFace datasets library
- **torch** (>=2.1.0) - PyTorch deep learning framework
- **numpy** (>=1.24.0) - Numerical computing
- **pandas** (>=2.0.0) - Data manipulation
- **matplotlib** (>=3.7.0) - Data visualization
- **seaborn** (>=0.12.0) - Statistical visualization
- **faiss-cpu** or **faiss-gpu** (>=1.7.0) - Efficient similarity search
- **tqdm** (>=4.65.0) - Progress bars

## 🎓 Topics Covered

### Semantic Search Fundamentals
- Understanding semantic vs keyword search
- Vector embeddings and semantic similarity
- Cosine similarity and distance metrics

### Text Embeddings
- Sentence transformers for semantic embeddings
- Mean pooling for sentence-level representations
- Batch processing for efficiency
- Embedding normalization techniques

### FAISS Index Types
- **IndexFlatIP**: Exact search with inner product
- **IndexFlatL2**: Exact search with L2 distance
- **IndexIVFFlat**: Approximate search with clustering
- Index selection based on scale and requirements

### Production Best Practices
- Saving and loading indices
- Incremental index updates
- GPU acceleration
- Hybrid search approaches
- Query preprocessing and optimization
- Performance monitoring and evaluation

## 📊 Dataset

This notebook uses the **lewtun/github-issues** dataset from HuggingFace Hub, which contains:
- GitHub issues from various repositories
- Issue titles, bodies, and metadata
- Perfect for demonstrating semantic search in technical documentation

## 🔗 Related Resources

### HuggingFace Resources
- [LLM Course Chapter 5.6](https://huggingface.co/learn/llm-course/chapter5/6?fw=pt) - Official semantic search tutorial
- [Sentence Transformers](https://www.sbert.net/) - Specialized models for semantic similarity
- [Datasets Hub](https://huggingface.co/datasets) - Explore more datasets

### FAISS Resources
- [FAISS Documentation](https://faiss.ai/) - Official FAISS documentation
- [FAISS GitHub](https://github.com/facebookresearch/faiss) - Source code and examples
- [FAISS Tutorial](https://github.com/facebookresearch/faiss/wiki) - Comprehensive guides

### Research Papers
- [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) - State-of-the-art retrieval
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) - Efficient sentence embeddings
- [ColBERT](https://arxiv.org/abs/2004.12832) - Contextualized late interaction

## 💡 Tips for Success

1. **Start with small datasets**: Use sampling to understand concepts before scaling
2. **Monitor GPU memory**: Use batch processing and gradient checkpointing for large models
3. **Experiment with models**: Try different sentence transformer models for your use case
4. **Measure search quality**: Track precision@k and user satisfaction metrics
5. **Use appropriate indices**: Match index type to your scale and accuracy requirements

## 🤝 Contributing

Found an issue or have suggestions for improvement? Please open an issue or submit a pull request to the main repository.

## 📄 License

This educational material is part of the HF Transformer Trove repository. Please refer to the main repository for license information.

## 👤 Author

**Vu Hung Nguyen** - AI Engineer & Researcher

Connect with me:
- 🌐 Website: [vuhung16au.github.io](https://vuhung16au.github.io/)
- 💼 LinkedIn: [linkedin.com/in/nguyenvuhung](https://www.linkedin.com/in/nguyenvuhung/)
- 💻 GitHub: [github.com/vuhung16au](https://github.com/vuhung16au/)

---

*Part of the [HF Transformer Trove](https://github.com/vuhung16au/hf-transformer-trove) educational series*
