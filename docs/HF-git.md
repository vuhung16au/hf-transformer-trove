# Git Integration with Hugging Face Hub

## 🎯 Learning Objectives
By the end of this document, you will understand:
- How to install and configure the Hugging Face CLI
- How to authenticate with Hugging Face Hub
- How to use git with git-lfs for managing large model files  
- Basic git workflow for Hugging Face repositories
- Best practices for version control with ML models and datasets

## 📋 Prerequisites
- Basic understanding of git version control concepts
- Familiarity with command line interface
- A Hugging Face Hub account (free at [huggingface.co](https://huggingface.co/join))
- Git installed on your system

## 📚 What We'll Cover
1. **Hugging Face CLI Setup**: Installation and authentication
2. **Git LFS Configuration**: Managing large files efficiently
3. **Basic Workflow**: Clone, modify, and push changes
4. **Repository Management**: Creating and updating HF repositories
5. **Best Practices**: Tips for effective version control with ML assets

---

## 1. Hugging Face CLI Setup

### Installation

The Hugging Face CLI provides seamless integration between git and the Hugging Face Hub.

```bash
# Install via pip (recommended)
pip install huggingface_hub

# Or install via Homebrew (macOS)
brew install huggingface-cli

# Or install via conda
conda install -c conda-forge huggingface_hub
```

### Authentication

Before you can push to Hugging Face repositories, you need to authenticate:

```bash
# Login with your Hugging Face credentials
huggingface-cli login

# Alternative: Use the shorter command
hf auth login
```

> 💡 **Pro Tip**: Your authentication token will be stored securely and used for all subsequent operations.

### Verify Installation

```bash
# Check CLI version and authentication status
huggingface-cli whoami

# Or use the shorter command
hf whoami
```

---

## 2. Git LFS Configuration

Hugging Face repositories use Git Large File Storage (LFS) to efficiently handle large model files, datasets, and other binary assets.

### What is Git LFS?

Git LFS replaces large files with text pointers inside Git, while storing the file contents on a remote server. This is essential for ML repositories containing:
- Model weights (`.bin`, `.safetensors` files)
- Datasets (`.arrow`, `.parquet` files)  
- Large configuration files
- Binary assets

### Install Git LFS

```bash
# Install git-lfs (if not already installed)
# Ubuntu/Debian
sudo apt install git-lfs

# macOS
brew install git-lfs

# Windows (via Git for Windows)
# Git LFS is typically included
```

### Initialize Git LFS

```bash
# Initialize git-lfs in your git configuration
git lfs install

# Verify installation
git lfs version
```

---

## 3. Basic Workflow

### Cloning a Repository

```bash
# Clone a Hugging Face repository
git clone https://huggingface.co/username/repository-name

# Example: Clone a model repository
git clone https://huggingface.co/cardiffnlp/twitter-roberta-base-hate-latest

# Navigate to the repository
cd repository-name
```

### Check Repository Status

```bash
# Check current status of your repository
git status

# See what files are tracked by LFS
git lfs ls-files

# Check remote configuration
git remote -v
```

### Making Changes

```bash
# Edit files locally (README.md, config.json, etc.)
# Add new model files, update documentation, etc.

# Stage your changes
git add .

# Or stage specific files
git add README.md config.json

# Check what will be committed
git status
```

### Committing and Pushing Changes

```bash
# Commit your changes with a descriptive message
git commit -m "Update model documentation and add new examples"

# Push changes to Hugging Face Hub
git push

# For the first push of a new branch
git push -u origin main
```

---

## 4. Repository Management

### Creating a New Repository

```bash
# Create a new repository on Hugging Face Hub
huggingface-cli repo create your-repository-name

# Or specify the type (model, dataset, space)
huggingface-cli repo create your-model-name --type model

# Alternative using hf command
hf repo create your-repository-name
```

### Uploading Files

```bash
# Upload a single file
huggingface-cli upload username/repository-name ./local-file.txt

# Upload entire directory
huggingface-cli upload username/repository-name ./local-directory/

# Upload with the hf command (equivalent)
hf upload username/repository-name ./local-file.txt

# Example from the issue reference
hf upload vuhung/dummy .
```

### Pulling Latest Changes

```bash
# Pull latest changes from the Hub
git pull

# Pull changes from a specific branch
git pull origin main

# Handle any merge conflicts if they arise
# Edit conflicted files, then:
git add .
git commit -m "Resolve merge conflicts"
```

---

## 5. Best Practices

### Repository Organization

```bash
# Typical Hugging Face repository structure
your-model/
├── README.md                 # Model card and documentation
├── config.json              # Model configuration
├── pytorch_model.bin         # Model weights (tracked by LFS)
├── tokenizer.json           # Tokenizer configuration
├── tokenizer_config.json    # Tokenizer settings
└── .gitattributes          # LFS file patterns
```

### Git LFS File Patterns

Ensure your `.gitattributes` file includes common ML file patterns:

```bash
# View current LFS patterns
cat .gitattributes

# Common patterns for ML repositories:
# *.bin filter=lfs diff=lfs merge=lfs -text
# *.safetensors filter=lfs diff=lfs merge=lfs -text
# *.arrow filter=lfs diff=lfs merge=lfs -text
# *.parquet filter=lfs diff=lfs merge=lfs -text
```

### Commit Message Conventions

```bash
# Use clear, descriptive commit messages
git commit -m "Add fine-tuned model for hate speech detection"
git commit -m "Update README with training details and examples"
git commit -m "Fix tokenizer configuration for proper inference"
```

### Branch Management

```bash
# Create a new branch for experimental changes
git checkout -b experiment/new-architecture

# Switch between branches
git checkout main
git checkout experiment/new-architecture

# Merge changes back to main
git checkout main
git merge experiment/new-architecture
```

---

## 💡 Pro Tips

> **🚀 Performance**: Use `git lfs pull` to download large files only when needed, especially useful for large model repositories.

```bash
# Clone repository without downloading LFS files
git clone --filter=blob:none https://huggingface.co/username/repository-name

# Download LFS files when needed
git lfs pull
```

> **💡 Pro Tip**: Use `.hfignore` file (similar to `.gitignore`) to exclude files from Hugging Face Hub uploads.

```bash
# Create .hfignore file
echo "__pycache__/" >> .hfignore
echo "*.pyc" >> .hfignore
echo ".DS_Store" >> .hfignore
```

> **⚠️ Common Pitfall**: Always ensure git-lfs is installed and initialized before working with model repositories to avoid corrupted large files.

---

## ⚠️ Common Pitfalls

### Large File Issues
- **Problem**: Pushing large files without LFS causes repository bloat
- **Solution**: Ensure `.gitattributes` properly tracks large file types with LFS

### Authentication Errors  
- **Problem**: Push rejected due to authentication issues
- **Solution**: Re-run `huggingface-cli login` and verify with `huggingface-cli whoami`

### Merge Conflicts
- **Problem**: Conflicts when multiple contributors edit the same files
- **Solution**: Use `git pull` before making changes and communicate with team members

---

## 🔗 Next Steps

### Related Documentation
- **[Checkpoints Guide](./checkpoints.md)**: Understanding model checkpoints and versioning
- **[Best Practices](./best-practices.md)**: Hugging Face ecosystem best practices

### External Resources
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [Git LFS Documentation](https://git-lfs.github.io/)
- [Hugging Face CLI Reference](https://huggingface.co/docs/huggingface_hub/guides/cli)

---

> **Key Takeaway**: The Hugging Face CLI bridges the gap between traditional git workflows and modern ML development, making it easy to version control models, datasets, and experiments while leveraging git-lfs for efficient large file management.