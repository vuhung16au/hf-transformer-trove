# Hugging Face Key Terms and Concepts

## Core Library Concepts

### AutoModel Classes
**Definition**: Automatic model selection classes that load the appropriate architecture based on model configuration.

```python
from transformers import AutoModel, AutoTokenizer

# Automatically loads the correct model architecture
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Key Classes**:
- `AutoModel`: Base transformer model
- `AutoModelForSequenceClassification`: For classification tasks
- `AutoModelForQuestionAnswering`: For Q&A tasks
- `AutoModelForMaskedLM`: For masked language modeling

### Model Hub
**Definition**: Central repository for pre-trained models, datasets, and spaces.

```python
from transformers import pipeline

# Load a model directly from the hub
classifier = pipeline("sentiment-analysis", 
                     model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Use the model
result = classifier("I love using Hugging Face!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

### Pipeline
**Definition**: High-level API for quick inference on common NLP tasks.

```python
from transformers import pipeline

# Text classification pipeline
classifier = pipeline("text-classification")
result = classifier("This movie is amazing!")

# Question answering pipeline
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company focused on NLP."
question = "What does Hugging Face focus on?"
answer = qa_pipeline(question=question, context=context)
```

## Tokenization Concepts

### Tokenizer
**Definition**: Converts text into numerical tokens that models can process.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenize text
text = "Hello, world!"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', ',', 'world', '!']

# Convert to input IDs
input_ids = tokenizer.encode(text)
print(input_ids)  # [101, 7592, 1010, 2088, 999, 102]

# Batch processing with padding
texts = ["Short text", "This is a much longer text that needs padding"]
batch = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
```

### Special Tokens
**Definition**: Special markers used by models (CLS, SEP, PAD, UNK, MASK).

```python
# Special tokens in BERT
print(tokenizer.cls_token)  # [CLS]
print(tokenizer.sep_token)  # [SEP]
print(tokenizer.pad_token)  # [PAD]
print(tokenizer.mask_token)  # [MASK]

# Adding special tokens
text = "Question: What is NLP? Answer: Natural Language Processing"
tokens = tokenizer.tokenize(text, add_special_tokens=True)
```

### Subword Tokenization
**Definition**: Breaking words into smaller meaningful units (BPE, WordPiece, SentencePiece).

```python
# Example with different tokenizers
from transformers import GPT2Tokenizer, BertTokenizer

# BPE (GPT-2)
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
print(gpt2_tokenizer.tokenize("unhappiness"))  # ['un', 'happiness']

# WordPiece (BERT)  
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
print(bert_tokenizer.tokenize("unhappiness"))  # ['un', '##hap', '##pi', '##ness']
```

## Training and Fine-tuning

### Trainer API
**Definition**: High-level API for training and evaluating models.

```python
from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
```

### Fine-tuning
**Definition**: Adapting a pre-trained model to a specific task or domain.

```python
from transformers import AutoModelForSequenceClassification

# Load model for fine-tuning
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,  # 3-class classification
    classifier_dropout=0.1
)

# Freeze some layers (optional)
for param in model.bert.embeddings.parameters():
    param.requires_grad = False
```

## Advanced Techniques

### PEFT (Parameter Efficient Fine-Tuning)
**Definition**: Methods to fine-tune models with fewer trainable parameters.

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "value"]
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
```

### Attention Mechanism
**Definition**: Mechanism allowing models to focus on relevant parts of input.

```python
# Get attention weights
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions  # Tuple of attention weights

# Visualize attention (simplified)
import matplotlib.pyplot as plt
import numpy as np

# Last layer, first head attention weights
attention = attentions[-1][0, 0].detach().numpy()
plt.imshow(attention, cmap='Blues')
plt.title("Attention Weights")
plt.show()
```

### Pipeline Tasks
**Definition**: Supported task types in the pipeline API.

```python
# Available pipeline tasks
tasks = [
    "text-classification",
    "sentiment-analysis", 
    "question-answering",
    "summarization",
    "translation",
    "text-generation",
    "fill-mask",
    "token-classification",
    "zero-shot-classification",
    "conversational"
]

# Example usage
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
text = "Your long text here..."
summary = summarizer(text, max_length=100, min_length=50, do_sample=False)
```

## Datasets Library

### Dataset Loading
**Definition**: Loading and processing datasets efficiently.

```python
from datasets import load_dataset

# Load from Hugging Face Hub
dataset = load_dataset("imdb")

# Load local files
dataset = load_dataset("csv", data_files="train.csv")

# Load streaming dataset (for large datasets)
dataset = load_dataset("oscar", "unshuffled_deduplicated_en", streaming=True)
```

### Dataset Processing
**Definition**: Transforming and preparing data for training.

```python
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# Apply preprocessing
processed_dataset = dataset.map(preprocess_function, batched=True)

# Filter dataset
filtered_dataset = dataset.filter(lambda x: len(x["text"]) > 100)

# Train/test split
dataset = dataset.train_test_split(test_size=0.2)
```