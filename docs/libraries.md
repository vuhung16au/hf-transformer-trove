# Libraries Used in HF Transformer Trove

This document provides a comprehensive overview of all libraries used across the Jupyter notebooks in this repository, categorized by their purpose and use cases in the context of Hugging Face (HF) ecosystem and Natural Language Processing (NLP).

## Table of Contents

- [Hugging Face Libraries](#hugging-face-libraries)
- [Machine Learning & Deep Learning Libraries](#machine-learning--deep-learning-libraries)
- [Visualization Libraries](#visualization-libraries)
- [Data Processing Libraries](#data-processing-libraries)
- [Utility Libraries](#utility-libraries)
- [Basic Python Libraries](#basic-python-libraries)
- [Third-Party Libraries](#third-party-libraries)
- [Summary Statistics](#summary-statistics)

---

## Hugging Face Libraries

Core libraries from the Hugging Face ecosystem for working with transformers and NLP tasks.

### `datasets`

**Description:** HuggingFace library for accessing and processing NLP datasets

**Used in:** 14 notebooks

**Use Cases in Our Repository:**
    - Loading datasets from HuggingFace Hub
    - Data preprocessing and transformation
    - Dataset splitting (train/validation/test)
    - Custom dataset creation and caching
    - Batch processing and data collation

<details>
<summary>View all 14 notebooks using this library</summary>

- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `evaluate`

**Description:** HuggingFace library for model evaluation metrics

**Used in:** 4 notebooks

**Use Cases in Our Repository:**
    - Computing accuracy, precision, recall, F1-score
    - BLEU scores for translation tasks
    - ROUGE scores for summarization
    - Model performance benchmarking
    - Custom metric evaluation

<details>
<summary>View all 4 notebooks using this library</summary>

- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`

</details>

### `huggingface_hub`

**Description:** Python client for interacting with the HuggingFace Hub

**Used in:** 2 notebooks

**Use Cases in Our Repository:**
    - Uploading models and datasets to HuggingFace Hub
    - Managing repositories and model versions
    - Authentication and API key management
    - Downloading models and files from Hub
    - Repository metadata management

<details>
<summary>View all 2 notebooks using this library</summary>

- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`

</details>

### `peft`

**Description:** Parameter-Efficient Fine-Tuning library for large language models

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - LoRA (Low-Rank Adaptation) fine-tuning
    - QLoRA for quantized models
    - Prefix tuning and prompt tuning
    - Memory-efficient model adaptation
    - Reducing computational costs for fine-tuning

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/09_peft_lora_qlora.ipynb`

</details>

### `transformers`

**Description:** Core HuggingFace library for state-of-the-art Natural Language Processing

**Used in:** 40 notebooks

**Use Cases in Our Repository:**
    - Loading pre-trained transformer models (BERT, GPT, RoBERTa, etc.)
    - Tokenization and text preprocessing
    - Fine-tuning models for classification, NER, QA tasks
    - Inference pipelines for sentiment analysis, translation, summarization
    - Hate speech detection and text classification

<details>
<summary>View all 40 notebooks using this library</summary>

- `examples/01_intro_hf_transformers.ipynb`
- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/01-sentiment-analysis.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/04-mask-filling.ipynb`
- `examples/basic1.2/05-question-answering.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.2/07-summarization.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.4/masked-language-model.ipynb`
- `examples/basic1.4/question-answering.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.4/token-classification.ipynb`
- `examples/basic1.4/translation.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.6/machine-translation.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `trl`

**Description:** Transformer Reinforcement Learning library

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - RLHF (Reinforcement Learning from Human Feedback)
    - PPO (Proximal Policy Optimization) training
    - Reward modeling
    - Training language models with human preferences
    - Fine-tuning with reinforcement learning

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/10_llms_rlhf.ipynb`

</details>

---

## Machine Learning & Deep Learning Libraries

Core frameworks and libraries for building, training, and evaluating deep learning models.

### `flax`

**Description:** Neural network library built on JAX

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - JAX-based model training
    - Functional neural networks
    - Research and experimentation
    - Alternative to PyTorch models
    - High-performance inference

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic3.6/hf-nlp-flax.ipynb`

</details>

### `jax`

**Description:** JAX library for high-performance machine learning research

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Automatic differentiation
    - GPU/TPU acceleration
    - Numerical computing
    - Research experiments
    - Performance optimization

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic3.6/hf-nlp-flax.ipynb`

</details>

### `numpy`

**Description:** Fundamental package for numerical computing in Python

**Used in:** 29 notebooks

**Use Cases in Our Repository:**
    - Array and matrix operations
    - Mathematical functions and linear algebra
    - Random number generation
    - Data preprocessing and transformation
    - Statistical computations

<details>
<summary>View all 29 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/04-mask-filling.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.4/token-classification.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `sklearn`

**Description:** Scikit-learn machine learning library

**Used in:** 7 notebooks

**Use Cases in Our Repository:**
    - Classification metrics (accuracy, precision, recall)
    - Train/test splitting
    - Confusion matrix generation
    - Model evaluation and validation
    - Preprocessing utilities

<details>
<summary>View all 7 notebooks using this library</summary>

- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `torch`

**Description:** PyTorch deep learning framework

**Used in:** 39 notebooks

**Use Cases in Our Repository:**
    - Neural network model building and training
    - Tensor operations and automatic differentiation
    - GPU/TPU acceleration
    - Model optimization and inference
    - Custom loss functions and training loops

<details>
<summary>View all 39 notebooks using this library</summary>

- `examples/01_intro_hf_transformers.ipynb`
- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/01-sentiment-analysis.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/04-mask-filling.ipynb`
- `examples/basic1.2/05-question-answering.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.2/07-summarization.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.4/masked-language-model.ipynb`
- `examples/basic1.4/question-answering.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.4/token-classification.ipynb`
- `examples/basic1.4/translation.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.6/machine-translation.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

---

## Visualization Libraries

Libraries for creating plots, charts, and visualizations to understand model performance and data distributions.

### `matplotlib`

**Description:** Comprehensive plotting library for Python

**Used in:** 30 notebooks

**Use Cases in Our Repository:**
    - Creating training loss curves
    - Plotting accuracy metrics over epochs
    - Visualizing model performance
    - Data distribution plots
    - Confusion matrix heatmaps

<details>
<summary>View all 30 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.2/07-summarization.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `seaborn`

**Description:** Statistical data visualization library built on matplotlib

**Used in:** 23 notebooks

**Use Cases in Our Repository:**
    - Enhanced confusion matrix visualizations
    - Distribution plots for text length analysis
    - Correlation heatmaps
    - Statistical relationship visualization
    - Beautiful default styling

<details>
<summary>View all 23 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

---

## Data Processing Libraries

Libraries for data manipulation, cleaning, and preprocessing.

### `pandas`

**Description:** Data manipulation and analysis library

**Used in:** 26 notebooks

**Use Cases in Our Repository:**
    - Loading and saving CSV/Excel files
    - Data cleaning and preprocessing
    - Creating DataFrames from datasets
    - Statistical analysis of text data
    - Data aggregation and grouping

<details>
<summary>View all 26 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/04-mask-filling.ipynb`
- `examples/basic1.2/07-summarization.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/token-classification.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

---

## Utility Libraries

Supporting libraries for progress tracking, environment management, and other utilities.

### `codecarbon`

**Description:** Library for tracking carbon emissions from code

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Measuring carbon footprint of model training
    - Environmental impact tracking
    - Energy consumption monitoring
    - Sustainability reporting
    - Green AI practices

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic1.4/EmissionTracker.ipynb`

</details>

### `dotenv`

**Description:** Python-dotenv for loading environment variables

**Used in:** 9 notebooks

**Use Cases in Our Repository:**
    - Loading API keys from .env.local
    - Managing HuggingFace tokens
    - Secure credential management
    - Environment configuration
    - Separating secrets from code

<details>
<summary>View all 9 notebooks using this library</summary>

- `examples/01_intro_hf_transformers.ipynb`
- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`

</details>

### `requests`

**Description:** HTTP library for making API requests

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Downloading models and datasets
    - API interactions
    - Web scraping for data collection
    - Remote file downloads
    - HTTP POST/GET operations

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic2.8/vLLM/performance_comparison.ipynb`

</details>

### `tqdm`

**Description:** Progress bar library for loops and iterations

**Used in:** 5 notebooks

**Use Cases in Our Repository:**
    - Training progress visualization
    - Dataset processing progress
    - Batch iteration tracking
    - Time estimation for long operations
    - User feedback during inference

<details>
<summary>View all 5 notebooks using this library</summary>

- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`

</details>

---

## Basic Python Libraries

Standard Python libraries from the standard library used for common programming tasks.

### `collections`

**Description:** Container datatypes

**Used in:** 11 notebooks

**Use Cases in Our Repository:**
    - Counter for word frequency
    - defaultdict for grouping
    - OrderedDict for maintaining order
    - Named tuples for data structures
    - Deque for efficient queues

<details>
<summary>View all 11 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `datetime`

**Description:** Date and time manipulation

**Used in:** 2 notebooks

**Use Cases in Our Repository:**
    - Timestamp generation
    - Training time logging
    - Date formatting
    - Time calculations
    - Experiment tracking

<details>
<summary>View all 2 notebooks using this library</summary>

- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`

</details>

### `html`

**Description:** HTML parsing and manipulation

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - HTML entity decoding
    - Text cleaning from web data
    - Special character handling
    - Web scraping preprocessing
    - Unicode normalization

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `json`

**Description:** JSON encoding and decoding

**Used in:** 10 notebooks

**Use Cases in Our Repository:**
    - Loading model configurations
    - Saving training results
    - Dataset metadata handling
    - API response parsing
    - Configuration file management

<details>
<summary>View all 10 notebooks using this library</summary>

- `examples/03_datasets_library.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `os`

**Description:** Operating system interface

**Used in:** 16 notebooks

**Use Cases in Our Repository:**
    - File path operations
    - Environment variable access
    - Directory creation and management
    - File system navigation
    - Cross-platform compatibility

<details>
<summary>View all 16 notebooks using this library</summary>

- `examples/01_intro_hf_transformers.ipynb`
- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `pathlib`

**Description:** Object-oriented filesystem paths

**Used in:** 3 notebooks

**Use Cases in Our Repository:**
    - Path operations
    - File existence checking
    - Directory traversal
    - Cross-platform path handling
    - Modern path manipulation

<details>
<summary>View all 3 notebooks using this library</summary>

- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`

</details>

### `pickle`

**Description:** Python object serialization

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Saving trained models
    - Caching preprocessed data
    - Object persistence
    - State saving
    - Quick data storage

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic5.2/working-with-datasets.ipynb`

</details>

### `random`

**Description:** Generate pseudo-random numbers

**Used in:** 3 notebooks

**Use Cases in Our Repository:**
    - Random seed setting (seed=16)
    - Data sampling
    - Shuffling datasets
    - Reproducible experiments
    - Random selection

<details>
<summary>View all 3 notebooks using this library</summary>

- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `re`

**Description:** Regular expression operations

**Used in:** 6 notebooks

**Use Cases in Our Repository:**
    - Text cleaning and preprocessing
    - Pattern matching
    - String extraction
    - Text normalization
    - URL and email parsing

<details>
<summary>View all 6 notebooks using this library</summary>

- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `shutil`

**Description:** High-level file operations

**Used in:** 2 notebooks

**Use Cases in Our Repository:**
    - File copying
    - Directory removal
    - Archive creation
    - Disk usage checking
    - File permissions

<details>
<summary>View all 2 notebooks using this library</summary>

- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`

</details>

### `subprocess`

**Description:** Subprocess management

**Used in:** 2 notebooks

**Use Cases in Our Repository:**
    - Running shell commands
    - External tool integration
    - System command execution
    - Pipeline orchestration
    - Process management

<details>
<summary>View all 2 notebooks using this library</summary>

- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`

</details>

### `sys`

**Description:** System-specific parameters and functions

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Python version checking
    - Command-line arguments
    - Exit codes
    - System path manipulation
    - Standard I/O access

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic3.6/hf-nlp-flax.ipynb`

</details>

### `tempfile`

**Description:** Generate temporary files and directories

**Used in:** 2 notebooks

**Use Cases in Our Repository:**
    - Temporary file creation
    - Cache management
    - Intermediate data storage
    - Testing utilities
    - Cleanup automation

<details>
<summary>View all 2 notebooks using this library</summary>

- `examples/03_datasets_library.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`

</details>

### `time`

**Description:** Time access and conversions

**Used in:** 30 notebooks

**Use Cases in Our Repository:**
    - Training duration measurement
    - Performance benchmarking
    - Timeout implementation
    - Timestamp generation
    - Speed comparisons

<details>
<summary>View all 30 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/01-sentiment-analysis.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/04-mask-filling.ipynb`
- `examples/basic1.2/07-summarization.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.4/token-classification.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.6/machine-translation.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `typing`

**Description:** Support for type hints

**Used in:** 23 notebooks

**Use Cases in Our Repository:**
    - Type annotations for functions
    - Code documentation
    - IDE autocomplete support
    - Type checking
    - Better code readability

<details>
<summary>View all 23 notebooks using this library</summary>

- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/07-summarization.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.6/machine-translation.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

### `warnings`

**Description:** Warning control

**Used in:** 36 notebooks

**Use Cases in Our Repository:**
    - Suppressing deprecation warnings
    - Filtering library warnings
    - Clean notebook output
    - Debug mode configuration
    - Error handling

<details>
<summary>View all 36 notebooks using this library</summary>

- `examples/01_intro_hf_transformers.ipynb`
- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/08_question_answering.ipynb`
- `examples/09_peft_lora_qlora.ipynb`
- `examples/10_llms_rlhf.ipynb`
- `examples/basic1.2/01-sentiment-analysis.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/04-mask-filling.ipynb`
- `examples/basic1.2/05-question-answering.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic1.2/08-translation.ipynb`
- `examples/basic1.4/EmissionTracker.ipynb`
- `examples/basic1.4/encoder-decoder.ipynb`
- `examples/basic1.4/masked-language-model.ipynb`
- `examples/basic1.4/question-answering.ipynb`
- `examples/basic1.4/text-classification.ipynb`
- `examples/basic1.4/token-classification.ipynb`
- `examples/basic1.4/translation.ipynb`
- `examples/basic1.6/01-few-shot-learning.ipynb`
- `examples/basic1.6/02-reasoning.ipynb`
- `examples/basic1.6/grammar-correction.ipynb`
- `examples/basic1.6/machine-translation.ipynb`
- `examples/basic1.9/Compare-Transformers-Hatespeech.ipynb`
- `examples/basic2.3/HF-Transformer-Basic.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic2.8/vLLM/performance_comparison.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/HF-full-training-demo.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`

</details>

---

## Third-Party Libraries

Additional third-party libraries used for specific tasks and platform integrations.

### `GPUtil`

**Description:** GPU utilization monitoring

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - GPU memory tracking
    - GPU utilization monitoring
    - Multi-GPU management
    - Resource allocation
    - Performance optimization

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic2.8/vLLM/performance_comparison.ipynb`

</details>

### `concurrent`

**Description:** Concurrent execution utilities

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Parallel processing
    - Thread pool execution
    - Asynchronous operations
    - Multi-threading
    - Performance optimization

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic2.8/vLLM/performance_comparison.ipynb`

</details>

### `data`

**Description:** data library

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Specialized tasks

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/08_question_answering.ipynb`

</details>

### `dataclasses`

**Description:** Data classes

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Configuration objects
    - Type-safe data structures
    - Clean data containers
    - Automatic initialization
    - Readable code

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic1.6/02-reasoning.ipynb`

</details>

### `google`

**Description:** Google Colab utilities

**Used in:** 16 notebooks

**Use Cases in Our Repository:**
    - Loading secrets from Colab
    - Drive mounting
    - TPU access
    - Colab-specific features
    - Authentication in notebooks

<details>
<summary>View all 16 notebooks using this library</summary>

- `examples/01_intro_hf_transformers.ipynb`
- `examples/02_tokenizers.ipynb`
- `examples/03_datasets_library.ipynb`
- `examples/04_mini_project.ipynb`
- `examples/05_fine_tuning_trainer.ipynb`
- `examples/07_summarization.ipynb`
- `examples/basic1.2/02-zero-shot-classification.ipynb`
- `examples/basic1.2/03-text-generation.ipynb`
- `examples/basic1.2/06-feature-extraction.ipynb`
- `examples/basic2.5/multiple-sequences.ipynb`
- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic3.6/hf-nlp-flax.ipynb`
- `examples/basic4.3/manage-repo-model-hub.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`

</details>

### `hashlib`

**Description:** Secure hash and message digest algorithms

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Data integrity verification
    - Checksum generation
    - File hashing
    - Cache key generation
    - Unique identifiers

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/07_summarization.ipynb`

</details>

### `image`

**Description:** image library

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Specialized tasks

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/07_summarization.ipynb`

</details>

### `nltk`

**Description:** Natural Language Toolkit

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Tokenization
    - Stemming and lemmatization
    - POS tagging
    - Text preprocessing
    - Corpus access

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/07_summarization.ipynb`

</details>

### `optax`

**Description:** Gradient processing and optimization library for JAX

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - JAX-based optimization
    - Custom optimizers
    - Gradient transformations
    - Learning rate schedules
    - Research experiments

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic3.6/hf-nlp-flax.ipynb`

</details>

### `psutil`

**Description:** Process and system utilities

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Memory usage monitoring
    - CPU utilization tracking
    - System resource management
    - Performance profiling
    - Resource optimization

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic2.8/vLLM/performance_comparison.ipynb`

</details>

### `sentence_transformers`

**Description:** Sentence embeddings using transformers

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Semantic similarity computation
    - Sentence embeddings
    - Text clustering
    - Semantic search
    - Vector representations

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic1.2/06-feature-extraction.ipynb`

</details>

### `string`

**Description:** Common string operations

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - String constants
    - Text formatting
    - Character sets
    - String templates
    - Text processing

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/02_tokenizers.ipynb`

</details>

### `threading`

**Description:** Thread-based parallelism

**Used in:** 1 notebooks

**Use Cases in Our Repository:**
    - Concurrent data processing
    - Background tasks
    - Parallel inference
    - Multi-threaded operations
    - Performance improvements

<details>
<summary>View all 1 notebooks using this library</summary>

- `examples/basic2.8/vLLM/performance_comparison.ipynb`

</details>

### `torch_xla`

**Description:** PyTorch XLA for TPU support

**Used in:** 5 notebooks

**Use Cases in Our Repository:**
    - TPU device management
    - Distributed training on TPUs
    - XLA optimization
    - Google Cloud TPU access
    - High-performance computing

<details>
<summary>View all 5 notebooks using this library</summary>

- `examples/basic3.4/HF-full-training-demo.ipynb`
- `examples/basic4.3/push_to_hub_API_demo.ipynb`
- `examples/basic5.2/lambda-functions-recipe-NLP.ipynb`
- `examples/basic5.2/working-with-datasets.ipynb`
- `examples/basic5.3/HF-training-data-preparation.ipynb`

</details>

---

## Summary Statistics

### Library Categories Overview

| Category | Number of Libraries |
|----------|---------------------|
| Hugging Face Libraries | 6 |
| ML/DL Libraries | 5 |
| Visualization Libraries | 2 |
| Data Processing Libraries | 1 |
| Utility Libraries | 4 |
| Basic Python Libraries | 16 |
| Third-Party Libraries | 14 |
| **Total** | **48** |

### Most Frequently Used Libraries

Top 10 libraries by number of notebooks:

| Library | Category | Notebooks |
|---------|----------|-----------|
| `transformers` | Hugging Face Libraries | 40 |
| `torch` | ML/DL Libraries | 39 |
| `warnings` | Basic Python Libraries | 36 |
| `time` | Basic Python Libraries | 30 |
| `matplotlib` | Visualization Libraries | 30 |
| `numpy` | ML/DL Libraries | 29 |
| `pandas` | Data Processing Libraries | 26 |
| `typing` | Basic Python Libraries | 23 |
| `seaborn` | Visualization Libraries | 23 |
| `os` | Basic Python Libraries | 16 |

### Repository Focus

This repository emphasizes:

- **Hugging Face Ecosystem**: Primary focus on `transformers`, `datasets`, and related HF libraries
- **NLP Tasks**: Comprehensive coverage including sentiment analysis, hate speech detection, summarization, translation, and question answering
- **PyTorch**: Primary deep learning framework (used in 39 notebooks)
- **Educational Approach**: Extensive use of visualization libraries (matplotlib, seaborn) for understanding
- **Production Ready**: Includes tools for model deployment, monitoring, and carbon emission tracking

---

## About This Repository

**HF Transformer Trove** is an educational resource focused on learning the Hugging Face ecosystem and natural language processing. The repository serves as comprehensive study notes with practical implementations for ML practitioners and students.

### Key Learning Areas

- **Implementation with HF**: Primary framework and ecosystem for all implementations
- **NLP**: Core domain focus with comprehensive coverage of NLP tasks
- **Hate Speech Detection**: Emphasized application area for practical examples

### Repository Structure

- 45 Jupyter notebooks across 13 directories
- Progressive learning path from basics to advanced topics
- Real-world applications with hate speech detection focus
- Multi-platform support (local, Colab, SageMaker)

---

*This document was automatically generated from analyzing all Jupyter notebooks in the `examples/` directory.*