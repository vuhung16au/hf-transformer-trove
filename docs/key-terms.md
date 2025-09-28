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

### Model Card
**Definition**: A document designed to provide users with a complete, standardized, and transparent overview of a machine learning model's characteristics, limitations, training data, and intended use. It is essentially a technical specification sheet and ethical disclosure statement for an AI model.

Model Cards serve as comprehensive documentation that helps users understand what a model does, how it was trained, what data it was trained on, its performance metrics, potential biases, limitations, and recommended use cases.

```python
from huggingface_hub import model_info, hf_hub_download
import json

def explore_model_card(model_name):
    """
    Explore a model's card and metadata from Hugging Face Hub.
    
    Args:
        model_name: Name of the model on Hugging Face Hub
    """
    try:
        # Get model information
        info = model_info(model_name)
        
        print(f"📋 Model Card Information for: {model_name}")
        print("=" * 50)
        print(f"🏷️  Pipeline Task: {info.pipeline_tag}")
        print(f"📚 Library: {info.library_name}")
        print(f"💾 Downloads: {info.downloads:,}")
        print(f"👍 Likes: {info.likes}")
        print(f"🏪 Created: {info.created_at}")
        print(f"📝 Last Modified: {info.last_modified}")
        
        # Display tags (including language, license, etc.)
        if info.tags:
            print(f"🏷️  Tags: {', '.join(info.tags[:8])}")
        
        # Model card content (README.md)
        try:
            readme_path = hf_hub_download(
                repo_id=model_name, 
                filename="README.md",
                repo_type="model"
            )
            print(f"\n📄 Model Card Content Available: {readme_path}")
        except Exception:
            print(f"\n⚠️  No README.md found for detailed model card")
        
        # Model configuration details
        if hasattr(info, 'config') and info.config:
            print(f"\n⚙️  Model Configuration:")
            for key, value in info.config.items():
                if key in ['architectures', 'model_type', 'task_specific_params']:
                    print(f"   {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error exploring model card for {model_name}: {e}")

# Explore model cards for preferred hate speech detection models
preferred_models = [
    "cardiffnlp/twitter-roberta-base-hate-latest",
    "facebook/roberta-hate-speech-dynabench-r4-target",
    "unitary/toxic-bert"
]

for model in preferred_models:
    explore_model_card(model)
    print("\n" + "-" * 60 + "\n")
```

**Accessing Model Card Information Programmatically**:

```python
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import model_info

def get_comprehensive_model_info(model_name):
    """
    Get comprehensive model information from different sources.
    
    Args:
        model_name: Name of the model on Hugging Face Hub
    """
    print(f"🔍 Comprehensive Analysis: {model_name}")
    print("=" * 55)
    
    # 1. Hub metadata
    try:
        hub_info = model_info(model_name)
        print(f"📊 Hub Statistics:")
        print(f"   Downloads: {hub_info.downloads:,}")
        print(f"   Task: {hub_info.pipeline_tag}")
        print(f"   Library: {hub_info.library_name}")
    except Exception as e:
        print(f"❌ Hub info error: {e}")
    
    # 2. Model configuration
    try:
        config = AutoConfig.from_pretrained(model_name)
        print(f"\n⚙️  Model Architecture:")
        print(f"   Model Type: {config.model_type}")
        print(f"   Architectures: {getattr(config, 'architectures', 'N/A')}")
        if hasattr(config, 'num_labels'):
            print(f"   Number of Labels: {config.num_labels}")
        if hasattr(config, 'vocab_size'):
            print(f"   Vocabulary Size: {config.vocab_size:,}")
    except Exception as e:
        print(f"❌ Config error: {e}")
    
    # 3. Tokenizer information
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"\n🔤 Tokenizer Details:")
        print(f"   Tokenizer Type: {tokenizer.__class__.__name__}")
        print(f"   Vocab Size: {tokenizer.vocab_size:,}")
        print(f"   Special Tokens: {list(tokenizer.special_tokens_map.keys())}")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
    
    print(f"\n💡 Model Card Access:")
    print(f"   🌐 Direct URL: https://huggingface.co/{model_name}")
    print(f"   📱 In Code: huggingface_hub.model_info('{model_name}')")

# Example usage with hate speech detection model
get_comprehensive_model_info("cardiffnlp/twitter-roberta-base-hate-latest")
```

**Key Components of a Model Card**:

1. **Model Description**: Purpose, architecture, and task type
2. **Training Data**: Datasets used, data sources, preprocessing steps
3. **Performance Metrics**: Accuracy, F1-score, precision, recall on benchmark datasets
4. **Limitations**: Known issues, failure cases, out-of-scope use cases
5. **Bias and Fairness**: Known biases, fairness evaluations, ethical considerations
6. **Environmental Impact**: Carbon footprint, computational requirements
7. **Usage Guidelines**: Intended use cases, licensing, citation information

```python
# Practical example: Reading model card for hate speech detection
def analyze_hate_speech_model_card(model_name="cardiffnlp/twitter-roberta-base-hate-latest"):
    """
    Analyze model card information for hate speech detection model.
    Educational example showing how to interpret model cards.
    """
    print(f"🛡️  Hate Speech Detection Model Analysis")
    print(f"Model: {model_name}")
    print("=" * 60)
    
    # Check model info
    info = model_info(model_name)
    
    # Educational interpretation
    print("📋 Model Card Interpretation:")
    print(f"✅ Task Type: {info.pipeline_tag}")
    print(f"✅ Suitable for: Text classification, content moderation")
    
    if 'twitter' in model_name.lower():
        print(f"🐦 Domain: Optimized for social media text")
        print(f"💡 Best for: Twitter-like short text, informal language")
    
    if 'hate' in model_name.lower():
        print(f"🎯 Specialization: Hate speech detection")
        print(f"⚠️  Consider: Ethical implications, potential biases")
        print(f"📝 Recommendation: Review bias documentation in model card")
    
    print(f"\n🔗 Always check the full model card at:")
    print(f"   https://huggingface.co/{model_name}")
    
    return info

# Educational usage
model_info = analyze_hate_speech_model_card()
```

**Best Practices for Using Model Cards**:

> 💡 **Pro Tip**: Always read the model card before using any model in production to understand its limitations and appropriate use cases.

> ⚠️ **Important**: Model cards contain crucial information about biases, limitations, and ethical considerations - especially important for sensitive applications like hate speech detection.

> 🚀 **Performance**: Use model card metrics to compare different models for your specific task and dataset.

**Model Card Creation** (for custom models):

```python
# Template for creating your own model cards
model_card_template = """
# Model Card for {model_name}

## Model Description
- **Developed by:** [Your Name/Organization]
- **Model type:** [e.g., Text Classification]
- **Language(s):** [e.g., English]
- **Task:** [e.g., Hate Speech Detection]

## Intended Use
### Primary Use Cases
- Content moderation for social media platforms
- Automated toxicity detection in online comments

### Out-of-Scope Use
- Not intended for legal decision making
- Should not be used without human oversight

## Training Data
- **Dataset:** [e.g., davidson/hate_speech_offensive]
- **Size:** [e.g., 25,000 labeled examples]
- **Preprocessing:** [Describe preprocessing steps]

## Performance
- **Accuracy:** 0.XX
- **F1-Score:** 0.XX
- **Precision:** 0.XX
- **Recall:** 0.XX

## Limitations and Biases
[Describe known limitations and potential biases]

## Ethical Considerations
[Discuss ethical implications and fairness concerns]

## Usage Example
```python
from transformers import pipeline
classifier = pipeline("text-classification", model="{model_name}")
result = classifier("Example text")
```
"""

print("📝 Model Card Template for Hate Speech Detection:")
print(model_card_template.format(model_name="your-username/hate-speech-detector"))
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

### Learning Curves
**Definition**: Graphical representations that show how model performance changes during training, helping diagnose training issues and optimize hyperparameters.

Learning curves plot training metrics (loss, accuracy) against training steps/epochs for both training and validation sets. They are essential for understanding model behavior and preventing overfitting.

```python
import matplotlib.pyplot as plt
import numpy as np
from transformers import Trainer, TrainingArguments, TrainerCallback

class LearningCurveCallback(TrainerCallback):
    """Custom callback to track learning curves during training."""
    
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_accuracies = []
        self.steps = []
    
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """Collect metrics at each logging step."""
        if "train_loss" in logs:
            self.train_losses.append(logs["train_loss"])
            self.steps.append(state.global_step)
        
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_accuracies.append(logs.get("eval_accuracy", 0))
    
    def plot_curves(self, title="Learning Curves - Hate Speech Detection"):
        """Plot comprehensive learning curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Training and Validation Loss
        axes[0, 0].plot(self.steps, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        if self.eval_losses:
            eval_steps = self.steps[-len(self.eval_losses):]
            axes[0, 0].plot(eval_steps, self.eval_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation Accuracy
        if self.eval_accuracies:
            eval_steps = self.steps[-len(self.eval_accuracies):]
            axes[0, 1].plot(eval_steps, self.eval_accuracies, 'g-', linewidth=2)
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Training Steps')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss Gap Analysis (Overfitting Detection)
        if self.eval_losses:
            eval_steps = self.steps[-len(self.eval_losses):]
            train_losses_aligned = self.train_losses[-len(self.eval_losses):]
            gap = np.array(self.eval_losses) - np.array(train_losses_aligned)
            axes[1, 0].plot(eval_steps, gap, 'orange', linewidth=2)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Overfitting Detection\n(Val Loss - Train Loss)')
            axes[1, 0].set_xlabel('Training Steps')
            axes[1, 0].set_ylabel('Loss Gap')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate Schedule (if available)
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nSchedule\n(Add LR tracking)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, style='italic')
        axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.show()
        
        # Print interpretation
        self._interpret_curves()
    
    def _interpret_curves(self):
        """Provide educational interpretation of the curves."""
        print("\n📊 Learning Curves Interpretation Guide:")
        print("=" * 50)
        
        if self.eval_losses and self.train_losses:
            latest_train_loss = self.train_losses[-1]
            latest_val_loss = self.eval_losses[-1]
            gap = latest_val_loss - latest_train_loss
            
            print(f"🔍 Current Status:")
            print(f"   Training Loss: {latest_train_loss:.4f}")
            print(f"   Validation Loss: {latest_val_loss:.4f}")
            print(f"   Gap: {gap:.4f}")
            
            if gap > 0.1:
                print("\n⚠️ Signs of Overfitting:")
                print("   • Validation loss higher than training loss")
                print("   • Consider: Early stopping, regularization, more data")
            elif gap < -0.05:
                print("\n🤔 Possible Underfitting:")
                print("   • Model may not be learning effectively")
                print("   • Consider: Lower regularization, more training")
            else:
                print("\n✅ Good Training Balance:")
                print("   • Training and validation losses are aligned")
        
        print(f"\n💡 General Tips:")
        print(f"   • Smooth decreasing curves = good learning")
        print(f"   • Plateauing = may need learning rate adjustment")
        print(f"   • Oscillating curves = learning rate too high")
        print(f"   • Increasing validation loss = overfitting")

# Example usage with hate speech detection
from datasets import load_dataset

# Load preferred hate speech dataset
dataset = load_dataset("tdavidson/hate_speech_offensive", split="train[:1000]")

# Initialize learning curve callback
learning_callback = LearningCurveCallback()

# Training arguments with frequent evaluation
training_args = TrainingArguments(
    output_dir="./hate_speech_model",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=100,  # Evaluate every 100 steps for detailed curves
    logging_steps=50,   # Log every 50 steps
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy",
)

# Add callback to trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset.select(range(200)),  # Small eval set for demo
    callbacks=[learning_callback],
)

# After training, plot learning curves
# trainer.train()
# learning_callback.plot_curves("Hate Speech Detection Learning Curves")
```

**Common Learning Curve Patterns**:

1. **Healthy Learning**: Both curves decrease smoothly, with validation slightly above training
2. **Overfitting**: Training loss continues decreasing while validation loss increases
3. **Underfitting**: Both curves plateau at high loss values
4. **High Variance**: Large gap between training and validation performance
5. **High Bias**: Both curves converge to suboptimal performance

**Practical Tips**:
- Plot curves frequently during training to catch issues early
- Use validation curves to determine optimal stopping point
- Compare multiple runs to ensure reproducibility
- Monitor both loss and task-specific metrics (accuracy, F1-score)
- Use curves to tune hyperparameters like learning rate and batch size

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