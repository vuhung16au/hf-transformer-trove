# Hugging Face Best Practices

## Model Loading and Management

### Efficient Model Loading
```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load model with specific precision to save memory
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    torch_dtype=torch.float16,  # Use half precision
    device_map="auto",  # Automatically distribute across GPUs
    low_cpu_mem_usage=True  # Reduce CPU memory usage during loading
)

# Load only tokenizer if you only need tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

### Model Caching
```python
import os
from transformers import AutoModel

# Set custom cache directory
os.environ["TRANSFORMERS_CACHE"] = "/path/to/custom/cache"

# Or use cache_dir parameter
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    cache_dir="/path/to/custom/cache"
)

# Clear cache when needed
from transformers.utils.hub import scan_cache_dir
cache_info = scan_cache_dir()
# Delete specific models if needed
```

## Tokenization Best Practices

### Batch Processing
```python
# Efficient batch tokenization
texts = ["Text 1", "Text 2", "Text 3"]

# Good: Process all at once
batch = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Avoid: Individual tokenization in loops
# for text in texts:  # Less efficient
#     tokenizer(text, ...)
```

### Dynamic Padding
```python
from transformers import DataCollatorWithPadding

# Use data collator for dynamic padding during training
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# This is more memory efficient than padding all sequences to max_length
```

### Handling Special Cases
```python
# Handle empty or very short texts
def safe_tokenize(text, tokenizer, max_length=512):
    if not text or len(text.strip()) == 0:
        text = "[EMPTY]"  # Placeholder for empty text
    
    return tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    )
```

## Training Best Practices

### Training Arguments Optimization
```python
from transformers import TrainingArguments

# Optimized training arguments
training_args = TrainingArguments(
    output_dir="./results",
    
    # Learning rate and optimization
    learning_rate=2e-5,  # Standard for BERT-like models
    adam_epsilon=1e-8,
    warmup_steps=500,
    weight_decay=0.01,
    
    # Batch size optimization
    per_device_train_batch_size=8,  # Adjust based on GPU memory
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
    
    # Training schedule
    num_train_epochs=3,
    max_steps=-1,  # Use epochs instead of steps
    
    # Evaluation and logging
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    
    # Model selection
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Reproducibility
    seed=42,
    data_seed=42,
    
    # Memory optimization
    fp16=True,  # Use mixed precision training
    dataloader_pin_memory=True,
    remove_unused_columns=True,
)
```

### Custom Training Loop
```python
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def custom_training_loop(model, train_dataloader, optimizer, scheduler, num_epochs):
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Track loss
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
```

## Memory and Performance Optimization

### Gradient Checkpointing
```python
# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Or set during model loading
model = AutoModel.from_pretrained(
    "bert-base-uncased",
    gradient_checkpointing=True
)
```

### Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler for mixed precision
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass with autocast
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss
    
    # Backward pass with scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Model Parallelism
```python
# For large models, use device mapping
model = AutoModel.from_pretrained(
    "microsoft/DialoGPT-large",
    device_map="auto",  # Automatically distribute across available GPUs
    torch_dtype=torch.float16
)

# Manual device mapping
device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.h.0": 0,
    "transformer.h.1": 0,
    "transformer.h.2": 1,
    "transformer.h.3": 1,
    "transformer.ln_f": 1,
    "lm_head": 1,
}
```

## Evaluation Best Practices

### Comprehensive Evaluation
```python
from evaluate import load
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Load multiple metrics
    accuracy_metric = load("accuracy")
    f1_metric = load("f1")
    
    # Get predictions
    predictions = np.argmax(predictions, axis=1)
    
    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }

# Use in trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
```

### Cross-validation
```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def cross_validate_model(dataset, model_name, num_folds=5):
    kfold = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset['text'], dataset['labels'])):
        print(f"Training fold {fold + 1}")
        
        # Split data
        train_dataset = dataset.select(train_idx)
        val_dataset = dataset.select(val_idx)
        
        # Train model
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        trainer = Trainer(...)  # Configure trainer
        
        trainer.train()
        eval_results = trainer.evaluate()
        scores.append(eval_results['eval_accuracy'])
    
    print(f"Cross-validation scores: {scores}")
    print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
```

## Error Handling and Debugging

### Common Issues and Solutions
```python
def robust_model_loading(model_name):
    try:
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    except OSError as e:
        print(f"Model not found: {model_name}")
        print("Suggestion: Check model name or internet connection")
        raise
    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Try reducing batch size or using CPU")
            # Retry with CPU
            model = AutoModel.from_pretrained(model_name, device_map="cpu")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        raise

def debug_tokenization(tokenizer, text):
    """Debug tokenization issues"""
    print(f"Original text: {text}")
    print(f"Tokens: {tokenizer.tokenize(text)}")
    print(f"Token IDs: {tokenizer.encode(text)}")
    print(f"Decoded: {tokenizer.decode(tokenizer.encode(text))}")
    
    # Check for special tokens
    special_tokens = {
        'CLS': tokenizer.cls_token_id,
        'SEP': tokenizer.sep_token_id,
        'PAD': tokenizer.pad_token_id,
        'UNK': tokenizer.unk_token_id,
    }
    print(f"Special tokens: {special_tokens}")
```

## Deployment Best Practices

### Model Optimization for Inference
```python
import torch

# Optimize model for inference
model.eval()  # Set to evaluation mode

# Use torch.no_grad() for inference
with torch.no_grad():
    outputs = model(**inputs)

# Convert to ONNX for faster inference
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size'}
    }
)
```

### Pipeline Optimization
```python
# Create optimized pipeline
pipe = pipeline(
    "text-classification",
    model="bert-base-uncased",
    tokenizer="bert-base-uncased",
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    batch_size=8,  # Process multiple texts at once
    max_length=512,
    truncation=True
)

# Batch processing
texts = ["Text 1", "Text 2", "Text 3", ...]
results = pipe(texts)  # Process all at once
```