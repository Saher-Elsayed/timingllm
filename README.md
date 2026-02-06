# TimingLLM: LLM-Augmented FPGA Timing Closure

[![Paper](https://img.shields.io/badge/Paper-ASPLOS%202026-blue)](https://arxiv.org/abs/xxxx.xxxxx)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://python.org)

> **LLM-Augmented FPGA Timing Closure: Toward Intelligent Static Timing Analysis Agents**
> 
> Saher Elsayed  
> University of Pennsylvania  
> Architecture 2.0 Workshop @ ASPLOS 2026

## Overview

**TimingLLM** is a framework that leverages Large Language Models (LLMs) augmented with Retrieval-Augmented Generation (RAG) to automate FPGA static timing analysis and timing closure. The system diagnoses timing violations, identifies root causes, and generates SDC constraint fixes automatically.

![TimingLLM Architecture](docs/architecture.png)

### Key Features

- 🎯 **82% F1 Score** on timing violation root cause classification
- ⚡ **100% Detection** for clock domain crossing and multicycle path violations
- 📉 **56% Average WNS Reduction** from automated constraint fixes
- 🔧 **RAG-Augmented Reasoning** with FPGA domain knowledge grounding

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/saherelsayed/timingllm.git
cd timingllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.timingllm_agent import TimingLLMAgent
from src.dataset_generator import generate_dataset

# Generate sample dataset
dataset = generate_dataset(num_designs=12)

# Initialize and train the agent
agent = TimingLLMAgent()
agent.train(training_paths)

# Diagnose a timing violation
result = agent.diagnose(timing_path)
print(f"Violation Type: {result.predicted_type}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Recommended Fix: {result.recommended_fix}")
```

### Run Experiments

```bash
# Run full experimental suite
python experiments/run_experiments.py

# Results will be saved to results/
```

## Project Structure

```
timingllm/
├── src/
│   ├── dataset_generator.py    # Synthetic timing violation generator
│   ├── timingllm_agent.py      # Core TimingLLM framework
│   └── utils.py                # Utility functions
├── experiments/
│   └── run_experiments.py      # Experiment runner
├── data/
│   └── timing_dataset.json     # Generated dataset
├── results/
│   ├── experiment_summary.json # Experimental results
│   └── figure_results.png      # Result visualizations
├── docs/
│   └── architecture.png        # Architecture diagram
├── requirements.txt
├── LICENSE
└── README.md
```

## Methodology

### 1. Domain Knowledge Base

TimingLLM uses three knowledge sources:

| Knowledge Type | Description | Size |
|---------------|-------------|------|
| Device Specifications | FPGA timing models, delays, routing | Device-specific |
| SDC Pattern Library | Curated constraint templates | 234 patterns |
| Historical Fix Database | Successful violation resolutions | 1,523 records |

### 2. RAG-Augmented Pipeline

1. **Context Extraction**: Parse STA reports for failing path details
2. **Knowledge Retrieval**: Query vector database for relevant patterns
3. **Feature Engineering**: Extract 10 timing-specific features
4. **Classification**: Random Forest predicts violation root cause
5. **Fix Generation**: Template-based SDC constraint generation

### 3. Violation Categories

| Category | Description | Detection F1 |
|----------|-------------|--------------|
| Clock Domain Crossing | Paths between async clocks | 1.00 |
| Multicycle Path | Multi-cycle operations | 1.00 |
| Architectural Bottleneck | High logic depth paths | 0.97 |
| Missing Constraint | Unconstrained paths | 0.70 |
| Hold Violation | Short path violations | 0.24 |

## Experimental Results

### Classification Performance

```
Overall Metrics (5-fold CV):
  Precision: 0.82
  Recall:    0.84
  F1 Score:  0.82 (±0.02)
```

### Slack Improvement

| Design Category | WNS Reduction | TNS Reduction |
|-----------------|---------------|---------------|
| Networking | 57% | 58% |
| Signal Processing | 62% | 61% |
| AI Accelerator | 49% | 58% |
| **Average** | **56%** | **59%** |

### Ablation Study

| Configuration | F1 Score | Degradation |
|--------------|----------|-------------|
| Full Model (with RAG) | 0.84 | — |
| Without RAG | 0.68 | 19% |

## Dataset

The synthetic dataset simulates realistic FPGA timing violations:

- **12 designs** across 3 application domains
- **658 timing violations** with ground truth labels
- **5 violation types** with realistic characteristics

### Design Categories

1. **Networking**: 100G Ethernet, PCIe Gen5, Switch Fabric, RDMA NIC
2. **Signal Processing**: FFT, MIMO Beamformer, Radar, Filter Bank
3. **AI Accelerator**: CNN, Transformer, GEMM, Quantized NN

## Citation

If you use TimingLLM in your research, please cite:

```bibtex
@inproceedings{elsayed2026timingllm,
  title={LLM-Augmented FPGA Timing Closure: Toward Intelligent Static Timing Analysis Agents},
  author={Elsayed, Saher},
  booktitle={Architecture 2.0 Workshop at ASPLOS},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work was conducted at the University of Pennsylvania.

## Contact

- **Saher Elsayed** - [selsayed@seas.upenn.edu](mailto:selsayed@seas.upenn.edu)
- **GitHub Issues** - For bug reports and feature requests
