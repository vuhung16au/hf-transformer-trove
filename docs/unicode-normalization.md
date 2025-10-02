# Unicode Normalization in Tokenization: A Comprehensive Guide

## 🎯 Learning Objectives
By the end of this document, you will understand:
- What Unicode normalization is and why it matters in NLP
- The four Unicode normalization forms (NFC, NFD, NFKC, NFKD)
- How normalization affects tokenization and model performance
- Practical implementation with Hugging Face tokenizers
- Best practices for text preprocessing with normalization
- Common normalization patterns in hate speech detection and NLP tasks

## 📋 Prerequisites
- Basic understanding of tokenization concepts
- Familiarity with Python and text processing
- Knowledge of Unicode and character encodings (helpful but not required)

## 📚 Table of Contents
- [What is Unicode Normalization?](#what-is-unicode-normalization)
- [The Four Normalization Forms](#the-four-normalization-forms)
- [Why Normalization Matters in NLP](#why-normalization-matters-in-nlp)
- [Unicode Normalization in Tokenization](#unicode-normalization-in-tokenization)
- [Implementing Normalization with Hugging Face](#implementing-normalization-with-hugging-face)
- [Normalization in Text Preprocessing](#normalization-in-text-preprocessing)
- [Best Practices and Common Pitfalls](#best-practices-and-common-pitfalls)
- [Advanced Topics](#advanced-topics)

## What is Unicode Normalization?

**Unicode normalization** is the process of transforming text into a canonical form where equivalent sequences of characters are represented in a consistent way. This is essential because Unicode allows multiple ways to represent the same character or text.

### The Problem: Multiple Representations

```python
import unicodedata

# Two ways to represent "café"
text1 = "café"  # Single character é (U+00E9)
text2 = "café"  # Base character e (U+0065) + combining accent ́ (U+0301)

print(f"Text 1: '{text1}'")
print(f"Text 2: '{text2}'")
print(f"Are they equal? {text1 == text2}")  # False!
print(f"Do they look the same? Yes!")

# Character breakdown
print(f"\nText 1 characters: {[hex(ord(c)) for c in text1]}")
print(f"Text 2 characters: {[hex(ord(c)) for c in text2]}")
```

> **Key Insight**: Without normalization, visually identical text may be treated as different strings, causing inconsistencies in tokenization, searching, and model training.

### Real-World Impact

```python
# Educational example showing why normalization matters
def demonstrate_normalization_impact():
    """
    Show how normalization affects text processing tasks.
    """
    # Text with different Unicode representations
    texts = [
        "naïve",  # Using precomposed characters
        "naïve",  # Using combining characters
        "résumé",
        "résumé",
    ]
    
    print("🔍 NORMALIZATION IMPACT DEMONSTRATION")
    print("=" * 50)
    
    # Without normalization
    print("\n❌ Without normalization:")
    unique_without = set(texts)
    print(f"Unique texts: {len(unique_without)}")
    for text in unique_without:
        print(f"  '{text}' ({len(text)} characters)")
    
    # With normalization
    print("\n✅ With normalization (NFC):")
    normalized_texts = [unicodedata.normalize('NFC', text) for text in texts]
    unique_with = set(normalized_texts)
    print(f"Unique texts: {len(unique_with)}")
    for text in unique_with:
        print(f"  '{text}' ({len(text)} characters)")

demonstrate_normalization_impact()
```

## The Four Normalization Forms

Unicode defines four normalization forms, each serving different purposes. Understanding these is crucial for proper text preprocessing in NLP tasks.

### 1. NFC (Normalization Form Canonical Composition)

**NFC** composes characters into their canonical (precomposed) forms.

```python
import unicodedata

def demonstrate_nfc():
    """
    Demonstrate NFC normalization - canonical composition.
    """
    print("📝 NFC: CANONICAL COMPOSITION")
    print("=" * 50)
    
    # Example 1: Combining accents
    text = "e\u0301"  # e + combining acute accent
    nfc_text = unicodedata.normalize('NFC', text)
    
    print(f"Original: '{text}' ({len(text)} characters)")
    print(f"  Characters: {[hex(ord(c)) for c in text]}")
    print(f"NFC:      '{nfc_text}' ({len(nfc_text)} character)")
    print(f"  Characters: {[hex(ord(c)) for c in nfc_text]}")
    
    # Example 2: Multiple accents
    examples = [
        ("naïve", "Word with umlaut"),
        ("café", "Word with acute accent"),
        ("Zürich", "City name with umlaut"),
        ("résumé", "Word with multiple accents"),
    ]
    
    print("\n📊 NFC Examples:")
    for text, description in examples:
        # Create decomposed version first
        decomposed = unicodedata.normalize('NFD', text)
        composed = unicodedata.normalize('NFC', decomposed)
        
        print(f"\n{description}:")
        print(f"  Decomposed: {len(decomposed)} chars → {[hex(ord(c)) for c in decomposed]}")
        print(f"  NFC:        {len(composed)} chars → {[hex(ord(c)) for c in composed]}")

demonstrate_nfc()
```

**When to use NFC:**
- ✅ General text processing and storage
- ✅ Most NLP applications and tokenization
- ✅ String comparison and matching
- ✅ Hate speech detection systems

### 2. NFD (Normalization Form Canonical Decomposition)

**NFD** decomposes characters into their canonical (base character + combining marks) forms.

```python
def demonstrate_nfd():
    """
    Demonstrate NFD normalization - canonical decomposition.
    """
    print("\n📝 NFD: CANONICAL DECOMPOSITION")
    print("=" * 50)
    
    # Example with precomposed characters
    text = "é"  # Precomposed e with acute (U+00E9)
    nfd_text = unicodedata.normalize('NFD', text)
    
    print(f"Original: '{text}' ({len(text)} character)")
    print(f"  Character: {hex(ord(text))}")
    print(f"NFD:      '{nfd_text}' ({len(nfd_text)} characters)")
    print(f"  Characters: {[hex(ord(c)) for c in nfd_text]}")
    print(f"  Breakdown: base 'e' + combining acute accent")
    
    # Example with complex text
    examples = [
        "Héllo Wörld",
        "Tôkyô",
        "Mötörhead",
        "Crème brûlée",
    ]
    
    print("\n📊 NFD Examples:")
    for text in examples:
        nfd = unicodedata.normalize('NFD', text)
        print(f"\n'{text}':")
        print(f"  Original: {len(text)} characters")
        print(f"  NFD:      {len(nfd)} characters")
        print(f"  Expansion: +{len(nfd) - len(text)} combining marks")

demonstrate_nfd()
```

**When to use NFD:**
- ✅ When you need to access base characters separately
- ✅ Analyzing text structure and diacritics
- ✅ Some search and indexing applications
- ✅ Working with combining character algorithms

### 3. NFKC (Normalization Form Compatibility Composition)

**NFKC** applies compatibility decomposition followed by canonical composition. It replaces compatibility characters with their equivalents.

```python
def demonstrate_nfkc():
    """
    Demonstrate NFKC normalization - compatibility composition.
    """
    print("\n📝 NFKC: COMPATIBILITY COMPOSITION")
    print("=" * 50)
    
    # Example showing compatibility transformations
    examples = [
        ("ﬁ", "fi", "Ligature fi"),  # U+FB01 → fi
        ("²", "2", "Superscript 2"),  # U+00B2 → 2
        ("℃", "°C", "Celsius symbol"),  # U+2103 → degree + C
        ("①", "1", "Circled digit 1"),  # U+2460 → 1
        ("ｈｅｌｌｏ", "hello", "Fullwidth Latin"),  # U+FF48... → h...
        ("½", "1⁄2", "Fraction one half"),  # U+00BD → 1/2 components
    ]
    
    print("🔄 NFKC Transformations:")
    for original, expected, description in examples:
        nfkc = unicodedata.normalize('NFKC', original)
        print(f"\n{description}:")
        print(f"  Original: '{original}' ({[hex(ord(c)) for c in original]})")
        print(f"  NFKC:     '{nfkc}' ({[hex(ord(c)) for c in nfkc]})")
        print(f"  Match:    {nfkc == expected or nfkc in expected}")
    
    # Real-world example for NLP
    messy_text = "The temperature is ２５℃ in Tôkyô!"
    clean_text = unicodedata.normalize('NFKC', messy_text)
    
    print(f"\n📊 Real-world NLP Example:")
    print(f"Original: '{messy_text}'")
    print(f"NFKC:     '{clean_text}'")
    print(f"Effect:   Normalizes fullwidth numbers and symbols")

demonstrate_nfkc()
```

**When to use NFKC:**
- ✅ Text preprocessing for machine learning models
- ✅ Search engines and information retrieval
- ✅ Social media text normalization
- ✅ **Hate speech detection** (preferred for consistency)
- ✅ When visual similarity should be treated as equivalence

> ⚠️ **Important**: NFKC may lose formatting information. Use with caution when preserving original formatting is important.

### 4. NFKD (Normalization Form Compatibility Decomposition)

**NFKD** applies compatibility decomposition, replacing compatibility characters and decomposing into base + combining.

```python
def demonstrate_nfkd():
    """
    Demonstrate NFKD normalization - compatibility decomposition.
    """
    print("\n📝 NFKD: COMPATIBILITY DECOMPOSITION")
    print("=" * 50)
    
    # Examples showing full decomposition
    examples = [
        "café",
        "℃",
        "ﬁnally",
        "²³",
        "Tôkyô",
    ]
    
    print("🔄 NFKD Full Decomposition:")
    for text in examples:
        nfkd = unicodedata.normalize('NFKD', text)
        
        print(f"\n'{text}':")
        print(f"  Original: {len(text)} chars → {[hex(ord(c)) for c in text]}")
        print(f"  NFKD:     {len(nfkd)} chars → {[hex(ord(c)) for c in nfkd]}")
        
        # Show decomposition details
        base_chars = [c for c in nfkd if unicodedata.category(c)[0] != 'M']
        combining = [c for c in nfkd if unicodedata.category(c)[0] == 'M']
        
        print(f"  Base characters: {len(base_chars)}")
        print(f"  Combining marks: {len(combining)}")

demonstrate_nfkd()
```

**When to use NFKD:**
- ✅ Maximum decomposition for analysis
- ✅ Stripping all diacritics and accents
- ✅ Text-to-speech preprocessing
- ✅ When you need both compatibility and decomposition

### Comparison of All Forms

```python
def compare_all_normalization_forms():
    """
    Compare all four normalization forms side by side.
    """
    print("\n📊 NORMALIZATION FORMS COMPARISON")
    print("=" * 80)
    
    test_cases = [
        "café",           # Accented character
        "ﬁle",           # Ligature
        "²",             # Superscript
        "℃",             # Composite symbol
        "naïve",         # Multiple accents
        "ｈｅｌｌｏ",      # Fullwidth
    ]
    
    forms = ['NFC', 'NFD', 'NFKC', 'NFKD']
    
    for text in test_cases:
        print(f"\nOriginal: '{text}' ({len(text)} chars)")
        print("-" * 60)
        
        for form in forms:
            normalized = unicodedata.normalize(form, text)
            print(f"{form:6}: '{normalized}' ({len(normalized)} chars)")
        
        print()

compare_all_normalization_forms()
```

## Why Normalization Matters in NLP

### 1. Consistent Tokenization

```python
from transformers import AutoTokenizer
import torch

# Set reproducible environment with repository standard seed=16
torch.manual_seed(16)
print("🔢 Using seed=16 for reproducibility (repository standard)")

def demonstrate_tokenization_impact():
    """
    Show how normalization affects tokenization results.
    """
    # Load tokenizer for hate speech detection (repository preference)
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
    
    print("\n🎯 NORMALIZATION IMPACT ON TOKENIZATION")
    print("=" * 60)
    
    # Example with different representations
    texts = [
        ("café", "NFC precomposed"),
        ("café", "NFD decomposed"),
    ]
    
    print("Without normalization:")
    for text, description in texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        print(f"\n  {description}:")
        print(f"    Text: '{text}' ({len(text)} chars)")
        print(f"    Tokens: {tokens}")
        print(f"    Token IDs: {token_ids}")
    
    # With normalization
    print("\n\n✅ With NFC normalization:")
    for text, description in texts:
        normalized = unicodedata.normalize('NFC', text)
        tokens = tokenizer.tokenize(normalized)
        token_ids = tokenizer.encode(normalized, add_special_tokens=False)
        
        print(f"\n  {description} → normalized:")
        print(f"    Text: '{normalized}' ({len(normalized)} chars)")
        print(f"    Tokens: {tokens}")
        print(f"    Token IDs: {token_ids}")

demonstrate_tokenization_impact()
```

### 2. Model Training Consistency

```python
def demonstrate_training_impact():
    """
    Show why normalization is crucial for model training.
    """
    print("\n📚 NORMALIZATION FOR MODEL TRAINING")
    print("=" * 60)
    
    # Simulate hate speech detection dataset
    training_examples = [
        ("This café serves great coffee!", "not_hate"),
        ("This café serves great coffee!", "not_hate"),  # Same text, different encoding
        ("naïve comment", "not_hate"),
        ("naïve comment", "not_hate"),  # Same text, different encoding
    ]
    
    print("❌ Without normalization:")
    print(f"  Examples that look identical: {len(training_examples)}")
    print(f"  Unique after set(): {len(set(ex[0] for ex in training_examples))}")
    print("  Problem: Duplicate processing, inconsistent features")
    
    print("\n✅ With NFC normalization:")
    normalized_examples = [
        (unicodedata.normalize('NFC', text), label) 
        for text, label in training_examples
    ]
    print(f"  Examples after normalization: {len(normalized_examples)}")
    print(f"  Unique after set(): {len(set(ex[0] for ex in normalized_examples))}")
    print("  Benefit: Consistent features, better model performance")

demonstrate_training_impact()
```

### 3. Search and Retrieval

```python
def demonstrate_search_impact():
    """
    Show how normalization improves search and matching.
    """
    print("\n🔍 NORMALIZATION FOR SEARCH & RETRIEVAL")
    print("=" * 60)
    
    # Simulate a document database
    documents = [
        "The naïve approach failed.",
        "Café culture in Paris.",
        "Résumé writing tips.",
        "Zürich travel guide.",
    ]
    
    # User search query (might be typed differently)
    queries = ["naive", "cafe", "resume", "Zurich"]
    
    print("Search without normalization:")
    for query in queries:
        matches = [doc for doc in documents if query.lower() in doc.lower()]
        print(f"  Query '{query}': {len(matches)} matches")
    
    print("\n✅ Search with NFKC normalization:")
    # Normalize both documents and queries
    norm_docs = [(unicodedata.normalize('NFKC', doc), doc) for doc in documents]
    
    for query in queries:
        norm_query = unicodedata.normalize('NFKC', query).lower()
        matches = [orig for norm, orig in norm_docs if norm_query in norm.lower()]
        print(f"  Query '{query}': {len(matches)} matches")
        for match in matches:
            print(f"    - {match}")

demonstrate_search_impact()
```

## Unicode Normalization in Tokenization

### Pre-tokenization Normalization

Modern tokenizers typically apply normalization as part of their preprocessing pipeline.

```python
from tokenizers import normalizers, Tokenizer, models

def explore_tokenizer_normalization():
    """
    Explore normalization options in tokenizer pipelines.
    """
    print("\n🔧 TOKENIZER NORMALIZATION PIPELINE")
    print("=" * 60)
    
    # Create a normalizer sequence
    normalizer_sequence = normalizers.Sequence([
        normalizers.NFD(),           # Decompose
        normalizers.Lowercase(),     # Lowercase
        normalizers.StripAccents(),  # Remove accents
    ])
    
    test_texts = [
        "CAFÉ",
        "Naïve",
        "Résumé",
        "Tôkyô",
    ]
    
    print("Normalization pipeline: NFD → Lowercase → StripAccents")
    for text in test_texts:
        # Simulate normalization steps
        nfd = unicodedata.normalize('NFD', text)
        lowered = nfd.lower()
        # Strip combining marks (accents)
        stripped = ''.join(c for c in lowered if unicodedata.category(c) != 'Mn')
        
        print(f"\n'{text}':")
        print(f"  NFD:    '{nfd}'")
        print(f"  Lower:  '{lowered}'")
        print(f"  Final:  '{stripped}'")

explore_tokenizer_normalization()
```

### BERT Tokenizer Normalization

```python
def analyze_bert_normalization():
    """
    Analyze how BERT tokenizer handles normalization.
    """
    from transformers import AutoTokenizer
    
    print("\n🤖 BERT TOKENIZER NORMALIZATION")
    print("=" * 60)
    
    # Load BERT tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    test_cases = [
        "HELLO WORLD",           # Uppercase
        "café",                   # Accented
        "ｈｅｌｌｏ",              # Fullwidth
        "Hello\u200bWorld",       # Zero-width space
        "  multiple   spaces  ",  # Whitespace
    ]
    
    print("BERT applies: Lowercase + NFD + StripAccents")
    for text in test_cases:
        tokens = tokenizer.tokenize(text)
        
        print(f"\nOriginal: '{text}'")
        print(f"Tokens:   {tokens}")

analyze_bert_normalization()
```

### RoBERTa and GPT Tokenizer Normalization

```python
def compare_tokenizer_normalization():
    """
    Compare normalization across different tokenizer types.
    """
    print("\n📊 TOKENIZER NORMALIZATION COMPARISON")
    print("=" * 60)
    
    tokenizers_config = {
        "BERT": "bert-base-uncased",
        "RoBERTa": "roberta-base",
        "GPT-2": "gpt2",
        "DistilBERT": "distilbert-base-uncased",
    }
    
    test_text = "Café Culture in Tôkyô"
    
    for name, model_name in tokenizers_config.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer.tokenize(test_text)
            
            print(f"\n{name}:")
            print(f"  Model: {model_name}")
            print(f"  Tokens: {tokens}")
            print(f"  Count: {len(tokens)} tokens")
            
        except Exception as e:
            print(f"\n{name}: Error loading - {e}")

compare_tokenizer_normalization()
```

## Implementing Normalization with Hugging Face

### Custom Preprocessing Pipeline

```python
import re

def create_preprocessing_pipeline():
    """
    Create a comprehensive preprocessing pipeline with normalization.
    Educational example for hate speech detection.
    """
    
    def preprocess_text(text, normalization_form='NFKC'):
        """
        Preprocess text with Unicode normalization.
        
        Args:
            text: Input text to preprocess
            normalization_form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
            
        Returns:
            Preprocessed text string
        """
        # Step 1: Unicode normalization
        text = unicodedata.normalize(normalization_form, text)
        
        # Step 2: Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Step 3: Remove zero-width characters
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\u200c', '')  # Zero-width non-joiner
        text = text.replace('\u200d', '')  # Zero-width joiner
        text = text.replace('\ufeff', '')  # Zero-width no-break space
        
        return text
    
    print("\n🔧 PREPROCESSING PIPELINE DEMONSTRATION")
    print("=" * 60)
    
    # Test cases with common issues
    test_texts = [
        "HATE   SPEECH  !!",           # Multiple spaces, uppercase
        "Naïve\u200bcomment",          # Zero-width space
        "ｈａｔｅ ｓｐｅｅｃｈ",        # Fullwidth
        "Café  culture\n\n",            # Mixed issues
        "This is ﬁne",                  # Ligature
    ]
    
    for original in test_texts:
        processed = preprocess_text(original)
        
        print(f"\nOriginal: '{repr(original)}'")
        print(f"Processed: '{processed}'")
        print(f"Length: {len(original)} → {len(processed)}")

create_preprocessing_pipeline()
```

### Hate Speech Detection Preprocessing

```python
def hate_speech_preprocessing():
    """
    Complete preprocessing pipeline for hate speech detection.
    Uses repository preferred model and seed=16.
    """
    import torch
    import numpy as np
    
    # Set reproducible environment with repository standard seed=16
    torch.manual_seed(16)
    np.random.seed(16)
    print("🔢 Random seed set to 16 for reproducibility (repository standard)")
    
    print("\n🛡️ HATE SPEECH DETECTION PREPROCESSING")
    print("=" * 60)
    
    # Load preferred hate speech detection model
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-hate-latest")
    
    def preprocess_for_hate_speech(text):
        """
        Preprocess text for hate speech detection.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed and normalized text
        """
        # 1. Unicode normalization (NFKC for social media)
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Lowercase (helps with generalization)
        text = text.lower()
        
        # 3. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 4. Remove zero-width characters
        text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
        
        # 5. Normalize common patterns
        text = text.replace("'", "'")  # Normalize apostrophes
        text = text.replace(""", '"').replace(""", '"')  # Normalize quotes
        
        return text
    
    # Example texts (educational examples)
    examples = [
        "This is a NORMAL comment!",
        "Café culture discussion",
        "Multiple   spaces   here",
        "Fullwidth: ｈｅｌｌｏ",
        "With\u200bZero\u200bWidth\u200bSpaces",
    ]
    
    print("Preprocessing pipeline:")
    print("  1. NFKC normalization")
    print("  2. Lowercase")
    print("  3. Whitespace normalization")
    print("  4. Remove zero-width characters")
    print("  5. Normalize apostrophes and quotes")
    
    for text in examples:
        processed = preprocess_for_hate_speech(text)
        tokens = tokenizer.tokenize(processed)
        
        print(f"\nOriginal:  '{text}'")
        print(f"Processed: '{processed}'")
        print(f"Tokens:    {tokens[:8]}{'...' if len(tokens) > 8 else ''}")

hate_speech_preprocessing()
```

### Custom Normalizer for Training

```python
def create_custom_normalizer():
    """
    Create a custom normalizer for tokenizer training.
    """
    from tokenizers import normalizers
    
    print("\n🏗️ CUSTOM NORMALIZER CONFIGURATION")
    print("=" * 60)
    
    # Define different normalizer strategies
    strategies = {
        "Minimal": normalizers.Sequence([
            normalizers.NFC(),
        ]),
        "Standard": normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
        ]),
        "Aggressive": normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.Lowercase(),
            normalizers.StripAccents(),
            normalizers.Replace(normalizers.Regex(r'\s+'), ' '),
        ]),
    }
    
    test_text = "Café Culture: Naïve Discussion"
    
    print(f"Test text: '{test_text}'")
    
    for strategy_name, normalizer in strategies.items():
        # Simulate normalization (actual API usage would be different)
        result = test_text
        
        print(f"\n{strategy_name} strategy:")
        print(f"  Configuration: {strategy_name}")
        
        # Manual simulation of the normalization steps
        if strategy_name == "Minimal":
            result = unicodedata.normalize('NFC', result)
        elif strategy_name == "Standard":
            result = unicodedata.normalize('NFKC', result)
            result = result.lower()
        elif strategy_name == "Aggressive":
            result = unicodedata.normalize('NFKC', result)
            result = result.lower()
            # Strip accents
            result = ''.join(c for c in unicodedata.normalize('NFD', result)
                           if unicodedata.category(c) != 'Mn')
            # Normalize whitespace
            result = re.sub(r'\s+', ' ', result)
        
        print(f"  Result: '{result}'")

create_custom_normalizer()
```

## Normalization in Text Preprocessing

### Comprehensive Preprocessing Function

```python
def comprehensive_text_preprocessing():
    """
    Complete text preprocessing with normalization for NLP tasks.
    """
    
    class TextPreprocessor:
        """
        Educational text preprocessor with Unicode normalization.
        """
        
        def __init__(self, normalization_form='NFKC', lowercase=True, 
                     strip_accents=False):
            """
            Initialize preprocessor with configuration.
            
            Args:
                normalization_form: Unicode normalization (NFC, NFD, NFKC, NFKD)
                lowercase: Whether to lowercase text
                strip_accents: Whether to remove accent marks
            """
            self.normalization_form = normalization_form
            self.lowercase = lowercase
            self.strip_accents = strip_accents
        
        def normalize_unicode(self, text):
            """Apply Unicode normalization."""
            return unicodedata.normalize(self.normalization_form, text)
        
        def normalize_whitespace(self, text):
            """Normalize whitespace to single spaces."""
            return re.sub(r'\s+', ' ', text).strip()
        
        def remove_zero_width(self, text):
            """Remove zero-width characters."""
            zero_width_chars = [
                '\u200b',  # Zero-width space
                '\u200c',  # Zero-width non-joiner
                '\u200d',  # Zero-width joiner
                '\ufeff',  # Zero-width no-break space
            ]
            for char in zero_width_chars:
                text = text.replace(char, '')
            return text
        
        def normalize_punctuation(self, text):
            """Normalize common punctuation variations."""
            # Normalize apostrophes
            text = text.replace("'", "'")
            text = text.replace("`", "'")
            
            # Normalize quotes
            text = text.replace(""", '"')
            text = text.replace(""", '"')
            text = text.replace("„", '"')
            
            # Normalize dashes
            text = text.replace("–", "-")  # En dash
            text = text.replace("—", "-")  # Em dash
            
            return text
        
        def strip_accent_marks(self, text):
            """Remove accent marks from characters."""
            if not self.strip_accents:
                return text
            
            # Decompose and remove combining marks
            nfd = unicodedata.normalize('NFD', text)
            return ''.join(c for c in nfd if unicodedata.category(c) != 'Mn')
        
        def preprocess(self, text):
            """
            Apply full preprocessing pipeline.
            
            Args:
                text: Input text string
                
            Returns:
                Preprocessed text
            """
            # 1. Unicode normalization
            text = self.normalize_unicode(text)
            
            # 2. Remove zero-width characters
            text = self.remove_zero_width(text)
            
            # 3. Normalize punctuation
            text = self.normalize_punctuation(text)
            
            # 4. Lowercase if enabled
            if self.lowercase:
                text = text.lower()
            
            # 5. Strip accents if enabled
            text = self.strip_accent_marks(text)
            
            # 6. Normalize whitespace
            text = self.normalize_whitespace(text)
            
            return text
    
    print("\n🔧 COMPREHENSIVE TEXT PREPROCESSING")
    print("=" * 60)
    
    # Test different configurations
    configs = [
        ("Minimal", TextPreprocessor('NFC', lowercase=False, strip_accents=False)),
        ("Standard", TextPreprocessor('NFKC', lowercase=True, strip_accents=False)),
        ("Aggressive", TextPreprocessor('NFKC', lowercase=True, strip_accents=True)),
    ]
    
    test_text = "Café\u200bCulture: Naïve Discussion in Tôkyô!"
    
    print(f"Original text: '{repr(test_text)}'")
    print(f"Visual:        '{test_text}'")
    
    for config_name, preprocessor in configs:
        result = preprocessor.preprocess(test_text)
        
        print(f"\n{config_name} configuration:")
        print(f"  Normalization: {preprocessor.normalization_form}")
        print(f"  Lowercase: {preprocessor.lowercase}")
        print(f"  Strip accents: {preprocessor.strip_accents}")
        print(f"  Result: '{result}'")

comprehensive_text_preprocessing()
```

### Integration with Dataset Processing

```python
def demonstrate_dataset_preprocessing():
    """
    Show how to apply normalization to datasets.
    """
    from datasets import Dataset
    import torch
    
    # Set reproducible environment with repository standard seed=16
    torch.manual_seed(16)
    print("🔢 Using seed=16 for reproducibility (repository standard)")
    
    print("\n📊 DATASET PREPROCESSING WITH NORMALIZATION")
    print("=" * 60)
    
    # Create sample dataset (simulating hate speech detection data)
    data = {
        'text': [
            "This is a normal comment",
            "SHOUTING IN ALL CAPS",
            "Café culture discussion",
            "Multiple   spaces   here",
            "Naïve\u200bwith\u200bzero\u200bwidth",
        ],
        'label': [0, 0, 0, 0, 0]  # 0 = not hate speech
    }
    
    dataset = Dataset.from_dict(data)
    
    def preprocess_function(examples):
        """
        Preprocessing function for dataset mapping.
        """
        processed_texts = []
        
        for text in examples['text']:
            # Apply NFKC normalization
            text = unicodedata.normalize('NFKC', text)
            
            # Lowercase
            text = text.lower()
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Remove zero-width characters
            text = re.sub(r'[\u200b-\u200d\ufeff]', '', text)
            
            processed_texts.append(text)
        
        return {'processed_text': processed_texts}
    
    # Apply preprocessing to dataset
    processed_dataset = dataset.map(preprocess_function, batched=True)
    
    print("Before and after preprocessing:")
    for i in range(len(dataset)):
        print(f"\n{i+1}. Original:  '{dataset[i]['text']}'")
        print(f"   Processed: '{processed_dataset[i]['processed_text']}'")

demonstrate_dataset_preprocessing()
```

## Best Practices and Common Pitfalls

### Best Practices

```python
def demonstrate_best_practices():
    """
    Show normalization best practices for NLP tasks.
    """
    print("\n✅ NORMALIZATION BEST PRACTICES")
    print("=" * 60)
    
    print("\n1. Choose the right normalization form:")
    print("   ✅ NFC: General text processing, storage")
    print("   ✅ NFKC: Machine learning, search, hate speech detection")
    print("   ✅ NFD: When you need to access base characters")
    print("   ✅ NFKD: Maximum decomposition for analysis")
    
    print("\n2. Apply normalization consistently:")
    print("   ✅ Normalize during both training and inference")
    print("   ✅ Document your normalization strategy")
    print("   ✅ Use the same form throughout your pipeline")
    
    print("\n3. Normalize early in the pipeline:")
    
    def good_pipeline(text):
        """Example of good preprocessing order."""
        text = unicodedata.normalize('NFKC', text)  # First
        text = text.lower()                          # Then lowercase
        text = re.sub(r'\s+', ' ', text)            # Then whitespace
        return text
    
    example = "Café\u200bCulture"
    result = good_pipeline(example)
    print(f"   ✅ Good: Normalize first → '{result}'")
    
    print("\n4. Handle edge cases:")
    edge_cases = [
        "\u200b",           # Zero-width space only
        "",                 # Empty string
        "   ",              # Whitespace only
        "\u0301",          # Combining mark only
    ]
    
    for text in edge_cases:
        try:
            normalized = unicodedata.normalize('NFKC', text)
            cleaned = normalized.strip()
            print(f"   ✅ '{repr(text)}' → '{repr(cleaned)}'")
        except Exception as e:
            print(f"   ❌ Error with '{repr(text)}': {e}")
    
    print("\n5. Test with diverse text:")
    print("   ✅ Test with multiple languages")
    print("   ✅ Test with special characters")
    print("   ✅ Test with user-generated content")
    print("   ✅ Test with historical or literary text")

demonstrate_best_practices()
```

### Common Pitfalls

```python
def demonstrate_common_pitfalls():
    """
    Show common mistakes with Unicode normalization.
    """
    print("\n❌ COMMON PITFALLS AND SOLUTIONS")
    print("=" * 60)
    
    print("\n1. Forgetting to normalize:")
    text1 = "café"
    text2 = "café"  # Different encoding
    
    print(f"   ❌ Without normalization: {text1 == text2}")
    
    norm1 = unicodedata.normalize('NFC', text1)
    norm2 = unicodedata.normalize('NFC', text2)
    print(f"   ✅ With normalization: {norm1 == norm2}")
    
    print("\n2. Normalizing too late:")
    print("   ❌ Bad: Tokenize first, then normalize")
    print("   ✅ Good: Normalize first, then tokenize")
    
    print("\n3. Using wrong normalization form:")
    original = "ﬁle"  # Ligature fi
    
    nfc = unicodedata.normalize('NFC', original)
    nfkc = unicodedata.normalize('NFKC', original)
    
    print(f"   Original: '{original}'")
    print(f"   NFC:  '{nfc}' (preserves ligature)")
    print(f"   NFKC: '{nfkc}' (expands ligature)")
    print("   ✅ Use NFKC for ML to avoid ligature issues")
    
    print("\n4. Inconsistent normalization:")
    print("   ❌ Training with NFKC, inference with NFC")
    print("   ✅ Use same normalization everywhere")
    
    print("\n5. Performance issues:")
    print("   ❌ Normalizing in tight loops")
    print("   ✅ Normalize once during preprocessing")
    
    # Performance demonstration
    import time
    
    texts = ["Sample text"] * 1000
    
    # Bad: Normalize repeatedly
    start = time.time()
    for text in texts:
        _ = unicodedata.normalize('NFKC', text)
        _ = unicodedata.normalize('NFKC', text)  # Redundant
    bad_time = time.time() - start
    
    # Good: Normalize once
    start = time.time()
    normalized = [unicodedata.normalize('NFKC', text) for text in texts]
    good_time = time.time() - start
    
    print(f"   Redundant normalization: {bad_time:.4f}s")
    print(f"   Single normalization: {good_time:.4f}s")

demonstrate_common_pitfalls()
```

### Testing Normalization

```python
def test_normalization_function():
    """
    Educational examples of testing normalization.
    """
    print("\n🧪 TESTING NORMALIZATION FUNCTIONS")
    print("=" * 60)
    
    def normalize_text(text, form='NFKC'):
        """Function to test."""
        return unicodedata.normalize(form, text)
    
    # Test cases with expected results
    test_cases = [
        # (input, form, expected)
        ("café", "NFC", "café"),
        ("café", "NFD", "cafe\u0301"),  # decomposed
        ("ﬁle", "NFKC", "file"),
        ("HELLO", "NFKC", "HELLO"),  # Unchanged
        ("  spaces  ", "NFKC", "  spaces  "),  # Whitespace preserved
    ]
    
    print("Running normalization tests:")
    passed = 0
    failed = 0
    
    for input_text, form, expected in test_cases:
        result = normalize_text(input_text, form)
        
        if result == expected:
            print(f"  ✅ PASS: '{input_text}' ({form}) → '{result}'")
            passed += 1
        else:
            print(f"  ❌ FAIL: '{input_text}' ({form})")
            print(f"     Expected: '{expected}'")
            print(f"     Got:      '{result}'")
            failed += 1
    
    print(f"\nResults: {passed} passed, {failed} failed")

test_normalization_function()
```

## Advanced Topics

### Performance Optimization

```python
def optimize_normalization_performance():
    """
    Show performance optimization techniques for normalization.
    """
    import time
    
    print("\n🚀 NORMALIZATION PERFORMANCE OPTIMIZATION")
    print("=" * 60)
    
    # Generate test data
    test_texts = ["Sample text with café and naïve"] * 10000
    
    # Technique 1: Batch processing
    print("1. Batch processing:")
    
    start = time.time()
    for text in test_texts:
        _ = unicodedata.normalize('NFKC', text)
    sequential_time = time.time() - start
    
    start = time.time()
    normalized_batch = [unicodedata.normalize('NFKC', text) for text in test_texts]
    batch_time = time.time() - start
    
    print(f"   Sequential: {sequential_time:.4f}s")
    print(f"   Batch:      {batch_time:.4f}s")
    
    # Technique 2: Caching
    print("\n2. Caching for repeated texts:")
    
    from functools import lru_cache
    
    @lru_cache(maxsize=1000)
    def cached_normalize(text):
        return unicodedata.normalize('NFKC', text)
    
    # Simulate repeated texts
    repeated_texts = ["text1", "text2", "text3"] * 3333
    
    start = time.time()
    for text in repeated_texts:
        _ = unicodedata.normalize('NFKC', text)
    uncached_time = time.time() - start
    
    start = time.time()
    for text in repeated_texts:
        _ = cached_normalize(text)
    cached_time = time.time() - start
    
    print(f"   Without cache: {uncached_time:.4f}s")
    print(f"   With cache:    {cached_time:.4f}s")
    print(f"   Speedup:       {uncached_time/cached_time:.1f}x")
    
    # Technique 3: Early filtering
    print("\n3. Skip normalization for ASCII-only text:")
    
    def smart_normalize(text):
        """Only normalize if text contains non-ASCII."""
        if all(ord(c) < 128 for c in text):
            return text  # Skip normalization for pure ASCII
        return unicodedata.normalize('NFKC', text)
    
    ascii_texts = ["Simple ASCII text"] * 10000
    
    start = time.time()
    for text in ascii_texts:
        _ = unicodedata.normalize('NFKC', text)
    always_time = time.time() - start
    
    start = time.time()
    for text in ascii_texts:
        _ = smart_normalize(text)
    smart_time = time.time() - start
    
    print(f"   Always normalize: {always_time:.4f}s")
    print(f"   Smart skip:       {smart_time:.4f}s")
    print(f"   Speedup:          {always_time/smart_time:.1f}x")

optimize_normalization_performance()
```

### Multilingual Normalization

```python
def multilingual_normalization():
    """
    Demonstrate normalization for multilingual text.
    """
    print("\n🌍 MULTILINGUAL TEXT NORMALIZATION")
    print("=" * 60)
    
    multilingual_examples = [
        ("English", "naïve café"),
        ("French", "château français"),
        ("German", "Zürich Müller"),
        ("Spanish", "niño señor"),
        ("Portuguese", "São Paulo"),
        ("Vietnamese", "Hà Nội"),  # Primary: English, Secondary: Vietnamese
        ("Japanese", "東京 Tōkyō"),  # Primary: English, Third: Japanese
    ]
    
    print("Normalization across languages:")
    
    for language, text in multilingual_examples:
        nfc = unicodedata.normalize('NFC', text)
        nfkc = unicodedata.normalize('NFKC', text)
        
        print(f"\n{language}:")
        print(f"  Original: '{text}' ({len(text)} chars)")
        print(f"  NFC:      '{nfc}' ({len(nfc)} chars)")
        print(f"  NFKC:     '{nfkc}' ({len(nfkc)} chars)")
        
        # Show if there are differences
        if nfc != nfkc:
            print(f"  Note: NFC and NFKC differ")

multilingual_normalization()
```

### Domain-Specific Normalization

```python
def domain_specific_normalization():
    """
    Show normalization strategies for different domains.
    """
    print("\n🎯 DOMAIN-SPECIFIC NORMALIZATION STRATEGIES")
    print("=" * 60)
    
    domains = {
        "Social Media": {
            "form": "NFKC",
            "lowercase": True,
            "strip_accents": False,
            "reason": "Handle fullwidth chars, emojis, preserve some styling"
        },
        "Academic Text": {
            "form": "NFC",
            "lowercase": False,
            "strip_accents": False,
            "reason": "Preserve original text formatting and accents"
        },
        "Search Engine": {
            "form": "NFKC",
            "lowercase": True,
            "strip_accents": True,
            "reason": "Maximum matching flexibility"
        },
        "Code/Technical": {
            "form": "NFC",
            "lowercase": False,
            "strip_accents": False,
            "reason": "Preserve exact representations"
        },
        "Hate Speech Detection": {
            "form": "NFKC",
            "lowercase": True,
            "strip_accents": False,
            "reason": "Handle social media text, normalize variants"
        },
    }
    
    for domain, config in domains.items():
        print(f"\n{domain}:")
        print(f"  Normalization form: {config['form']}")
        print(f"  Lowercase: {config['lowercase']}")
        print(f"  Strip accents: {config['strip_accents']}")
        print(f"  Reason: {config['reason']}")

domain_specific_normalization()
```

---

## 📋 Summary

### 🔑 Key Concepts Mastered

- **Unicode Normalization**: Understanding how to create consistent text representations
- **Four Normal Forms**: NFC, NFD, NFKC, NFKD and when to use each
- **Tokenization Impact**: How normalization affects token generation and model performance
- **Preprocessing Pipelines**: Integrating normalization into text preprocessing workflows
- **Performance Optimization**: Techniques for efficient normalization at scale

### 📈 Best Practices Learned

- **Normalize Early**: Apply normalization at the start of preprocessing pipeline
- **Be Consistent**: Use the same normalization form for training and inference
- **Choose Wisely**: Select appropriate normalization form for your use case (NFKC for ML)
- **Test Thoroughly**: Validate normalization with diverse text samples
- **Document Choices**: Always document your normalization strategy

### 🚀 Next Steps

- **Advanced Tokenization**: Explore [BPE](BPE.md) and [WordPiece](WordPiece.md) tokenization methods
- **Custom Tokenizers**: Learn to train custom tokenizers with normalization
- **Multilingual Models**: Study language-specific normalization challenges
- **Performance Tuning**: Advanced optimization for production systems
- **Hate Speech Detection**: Apply normalization in [hate speech detection models](../examples/)

---

## Further Reading

### Official Documentation
- [Unicode Technical Report #15: Normalization Forms](http://www.unicode.org/reports/tr15/)
- [Hugging Face Tokenizers - Normalization](https://huggingface.co/docs/tokenizers/api/normalizers)
- [Python unicodedata module](https://docs.python.org/3/library/unicodedata.html)

### Hugging Face Resources
- [LLM Course: Tokenization Chapter](https://huggingface.co/learn/llm-course/chapter6/4?fw=pt)
- [Tokenizers Documentation](https://huggingface.co/docs/tokenizers/)
- [Transformers Preprocessing](https://huggingface.co/docs/transformers/preprocessing)

### Related Documentation
- **[Tokenizers](Tokenizers.md)**: Complete guide to tokenization fundamentals
- **[BPE](BPE.md)**: Byte Pair Encoding tokenization
- **[WordPiece](WordPiece.md)**: WordPiece tokenization in BERT
- **[Best Practices](best-practices.md)**: General Hugging Face best practices

### Research Papers
- [Unicode Normalization Forms (Davis & Dürst, 2001)](http://www.unicode.org/reports/tr15/)
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)

---

## About the Author

**Vu Hung Nguyen** - AI Engineer & Researcher

Connect with me:
- 🌐 **Website**: [vuhung16au.github.io](https://vuhung16au.github.io/)
- 💼 **LinkedIn**: [linkedin.com/in/nguyenvuhung](https://www.linkedin.com/in/nguyenvuhung/)
- 💻 **GitHub**: [github.com/vuhung16au](https://github.com/vuhung16au/)

*This documentation is part of the [HF Transformer Trove](https://github.com/vuhung16au/hf-transformer-trove) educational series, designed to help you master the Hugging Face ecosystem through practical, well-explained examples.*
