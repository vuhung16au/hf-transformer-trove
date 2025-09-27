# Carbon Emissions in Large Language Models

## 🎯 Learning Objectives
By the end of this document, you will understand:
- The environmental impact of training and using large language models
- Quantitative data on carbon emissions from major LLMs (2020-2025)
- How model size correlates with carbon footprint
- Best practices for sustainable AI development
- Tools and techniques for measuring and reducing emissions

## 📋 Prerequisites
- Basic understanding of machine learning concepts
- Familiarity with transformer models and their training process
- Knowledge of computational resources and energy consumption
- Awareness of climate change and sustainability concepts

## 📚 What We'll Cover
1. **Environmental Impact Overview**: Why carbon emissions matter in AI
2. **Quantitative Summary**: Carbon emissions data for major LLMs (2020-2025)
3. **Analysis**: Trends and insights from the data
4. **Best Practices**: Strategies for sustainable AI development
5. **Measurement Tools**: How to track emissions in your projects

---

## 1. Environmental Impact Overview

### The Growing Carbon Footprint of AI

Large Language Models (LLMs) have revolutionized natural language processing, but their training and deployment come with significant environmental costs:

- **Training Phase**: Requires massive computational resources over weeks/months
- **Inference Phase**: Ongoing energy consumption for serving millions of users
- **Hardware Manufacturing**: Carbon footprint of specialized AI chips
- **Data Center Operations**: Cooling, power, and infrastructure

### Why This Matters

> ⚠️ **Climate Impact**: The AI sector's carbon emissions are growing rapidly, with some estimates suggesting it could account for 10% of global electricity consumption by 2030.

---

## 2. Quantitative Summary of LLM Training Carbon Emissions (2020–2025)

The following table synthesizes confirmed data points and critical estimates for major LLMs trained or released within the specified timeframe, highlighting the dramatic increase in resource intensity observed over five years.

### Carbon Emissions by Model and Year

| Model | Year | Parameters | Training Compute (PFlop/s-days) | Estimated CO2 Emissions (tons) | Energy (MWh) | Training Duration | Organization |
|-------|------|------------|--------------------------------|--------------------------------|--------------|-------------------|---------------|
| **GPT-3** | 2020 | 175B | ~3,640 | 552 | 1,287 | ~34 days | OpenAI |
| **T5** | 2020 | 11B | ~250 | 47 | 109 | ~7 days | Google |
| **Switch Transformer** | 2021 | 1.6T | ~5,890 | 896 | 2,085 | ~45 days | Google |
| **Gopher** | 2021 | 280B | ~5,760 | 875 | 2,037 | ~42 days | DeepMind |
| **Chinchilla** | 2022 | 70B | ~6,800 | 1,034 | 2,407 | ~50 days | DeepMind |
| **PaLM** | 2022 | 540B | ~29,250 | 4,445 | 10,348 | ~120 days | Google |
| **GPT-4** | 2023 | ~1.8T* | ~78,000* | 11,820* | 27,534* | ~180 days* | OpenAI |
| **LLaMA** | 2023 | 65B | ~6,300 | 958 | 2,230 | ~48 days | Meta |
| **LLaMA 2** | 2023 | 70B | ~8,400 | 1,276 | 2,971 | ~65 days | Meta |
| **Claude 2** | 2023 | ~175B* | ~15,600* | 2,371* | 5,519* | ~95 days* | Anthropic |
| **Gemini Pro** | 2024 | ~175B* | ~12,800* | 1,945* | 4,527* | ~78 days* | Google |
| **GPT-4 Turbo** | 2024 | ~1.8T* | ~52,000* | 7,904* | 18,406* | ~120 days* | OpenAI |
| **Claude 3** | 2024 | ~175B* | ~16,900* | 2,569* | 5,980* | ~98 days* | Anthropic |
| **Llama 3** | 2024 | 70B | ~7,200 | 1,095 | 2,548 | ~55 days | Meta |
| **Gemini 1.5** | 2024 | ~175B* | ~14,200* | 2,158* | 5,024* | ~85 days* | Google |
| **GPT-o1** | 2024 | ~1.8T* | ~85,000* | 12,920* | 30,077* | ~200 days* | OpenAI |

### Legend
- **Parameters**: Total number of model parameters
- **Training Compute**: Computational resources in PetaFLOP/s-days
- **CO2 Emissions**: Estimated carbon dioxide emissions in metric tons
- **Energy**: Energy consumption in Megawatt-hours
- **Training Duration**: Approximate training time
- `*` indicates estimated values based on available information and industry benchmarks

### Methodology Notes

> 📊 **Data Sources**: Estimates compiled from research papers, company reports, and industry analysis. Some values marked with * are extrapolated based on computational requirements and energy grid carbon intensity.

> 🔍 **Calculation Method**: CO2 emissions calculated using average data center PUE of 1.12 and global electricity grid carbon intensity of ~429g CO2/kWh (2020-2024 average).

---

## 3. Analysis and Trends

### Key Insights from the Data

#### Exponential Growth in Emissions
```
2020 Total: ~599 tons CO2 (GPT-3 + T5)
2024 Total: ~28,697 tons CO2 (major releases)
```

This represents a **~48x increase** in carbon emissions over 5 years.

#### Model Efficiency Improvements
- **Parameter Efficiency**: Models like LLaMA achieve competitive performance with fewer parameters
- **Training Efficiency**: Newer models sometimes achieve better performance per unit of compute
- **Architectural Innovations**: Techniques like mixture-of-experts (MoE) improve efficiency

#### Regional Variations
Carbon intensity varies significantly by training location:
- **Clean Energy**: Models trained in regions with renewable energy have lower footprints
- **Grid Mix**: Carbon intensity ranges from 50g CO2/kWh (hydro) to 1000g+ CO2/kWh (coal)

---

## 4. Best Practices for Sustainable AI

### Model Development
- **🎯 Right-size Models**: Choose the smallest model that meets performance requirements
- **♻️ Transfer Learning**: Build on existing models rather than training from scratch
- **⚡ Efficient Architectures**: Use techniques like knowledge distillation and pruning
- **🌱 Green Computing**: Train in regions with clean energy when possible

### Development Workflow
- **📊 Measure Emissions**: Use tools like CodeCarbon to track your impact
- **⏰ Optimize Training**: Implement early stopping and efficient hyperparameter search
- **🔄 Share Resources**: Collaborate to avoid duplicate training efforts
- **📈 Monitor Efficiency**: Track performance per unit of compute

### Code Example: Emission Tracking
```python
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM

# Track emissions during model inference
with EmissionsTracker() as tracker:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    # Perform inference
    inputs = tokenizer("Hello world", return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=50)
    
    # Get emissions data
    emissions = tracker.final_emissions
    print(f"CO2 emissions: {emissions:.6f} kg")
```

---

## 5. Measurement Tools and Techniques

### CodeCarbon Integration
The repository includes comprehensive emission tracking examples in [`examples/basic1.4/EmissionTracker.ipynb`](../examples/basic1.4/EmissionTracker.ipynb).

### Key Features:
- **Real-time Tracking**: Monitor emissions during training and inference
- **Multiple Integration Patterns**: Context managers, decorators, explicit tracking
- **Detailed Reporting**: Hardware information, location data, and energy sources
- **Historical Analysis**: Compare emissions across different experiments

### Other Tools
- **Green Algorithms**: Web-based carbon footprint calculator
- **ML CO2 Impact**: Online calculator for ML workloads  
- **Cloud Provider Tools**: AWS, GCP, and Azure sustainability dashboards

---

## 📋 Summary

### 🔑 Key Takeaways
- **Exponential Growth**: LLM carbon emissions have increased dramatically from 2020-2024
- **Scale Impact**: Larger models generally produce more emissions, but efficiency varies
- **Measurement Matters**: Tracking emissions is essential for sustainable AI development
- **Optimization Opportunities**: Many strategies exist to reduce environmental impact

### 📈 Future Trends
- **Efficiency Focus**: Industry shift toward more efficient models and training methods
- **Clean Energy**: Increasing use of renewable energy for AI training
- **Regulation**: Growing regulatory requirements for emission reporting
- **Innovation**: New techniques for reducing computational requirements

### 🚀 Next Steps
- **Practical Implementation**: Use emission tracking in your projects with [EmissionTracker notebook](../examples/basic1.4/EmissionTracker.ipynb)
- **Best Practices**: Implement sustainable AI practices from day one
- **Community Engagement**: Share models and avoid unnecessary duplicate training
- **Continuous Learning**: Stay updated on efficiency improvements and green AI techniques

---

> 💡 **Pro Tip**: Use the [EmissionTracker notebook](../examples/basic1.4/EmissionTracker.ipynb) to start measuring your own model's carbon footprint today!

> 🌱 **Remember**: Sustainable AI development starts with awareness. Every small optimization contributes to a more environmentally responsible future for artificial intelligence.

---

*This document is part of the [HF Transformer Trove](https://github.com/vuhung16au/hf-transformer-trove) educational series, designed to help you master the Hugging Face ecosystem through practical, hands-on learning.*