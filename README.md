# CSE472 - LLM Mediation Project

This project explores using Large Language Models (LLMs) as mediators in online conflicts, implementing mediation strategies, evaluation frameworks, and multi-agent debate systems.

## System Requirements

### Operating System
- **Linux** (tested on HPC environments with SLURM)
- GPU-enabled system recommended (NVIDIA A100 or similar)

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 24GB VRAM (A100 recommended)
- **RAM**: Minimum 32GB system memory
- **Storage**: At least 50GB free space for models and outputs

### Software Dependencies
- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (for GPU acceleration)
- **PyTorch**: 2.0.0 or higher

## Installation

### Install Python Dependencies
```bash
cd CSE472/scripts
pip install -r requirements.txt
```

**Required packages:**
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `huggingface-hub>=0.19.0`
- `accelerate>=0.24.0`

### Optional: Install Flash Attention 2 (for faster inference)
```bash
pip install flash-attn --no-build-isolation
```

## Directory Structure

```
CSE472/
├── README.md                          # This file
├── .gitignore                         # Git ignore configuration
├── Dataset_v2/                        # Input dataset (520 JSON files)
│   └── *.json                         # Conversation threads from Reddit
├── reddit.csv                         # Original Reddit data
├── output/                            # Generated outputs
│   ├── mediation_outputs.json         # Part 1 results
│   ├── evaluation_results.json        # Part 2 results
│   ├── advanced_evaluation_results.json  # Part 3 results
│   ├── user_sim_results_full.json     # Multi-agent debate results
│   └── combined_results.json          # Combined analysis
└── scripts/                           # Execution scripts
    ├── requirements.txt               # Python dependencies
    ├── part1_mediation.py             # Mediation generation
    ├── part2_evaluation.py            # Basic evaluation
    ├── part3_advanced_eval.py         # Advanced evaluation (few-shot & rubric)
    ├── part3_multi_agent_debate.py    # Multi-agent debate system
    ├── part5_comparative_analysis.py  # Comparative analysis
    └── cse472project2_part4_sol.py    # Additional solution script
```

### Script Execution Order

All scripts should be run from the `scripts/` directory:

```bash
cd /home/asrini81/CSE472/CSE472/scripts
```

#### Part 1: Generate Mediation Outputs
```bash
python part1_mediation.py
```

**What it does:**
- Loads conversations from `Dataset_v2/`
- Uses `meta-llama/Llama-3.2-3B-Instruct` model
- Generates judgment and steering messages for each conversation
- Outputs to `../output/mediation_outputs.json`

**Runtime:** ~30-60 minutes (depends on dataset size and GPU)

**Models downloaded:** ~6GB (Llama-3.2-3B-Instruct)

---

#### Part 2: Evaluate Mediation Quality
```bash
python part2_evaluation.py
```

**What it does:**
- Loads mediation outputs from Part 1
- Uses `Qwen/Qwen3-4B` as judge model
- Evaluates judgment and steering messages on clarity, fairness, and constructiveness
- Outputs to `../output/evaluation_results.json`

**Prerequisites:** Must run Part 1 first

**Runtime:** ~20-40 minutes

**Models downloaded:** ~8GB (Qwen3-4B)

---

#### Part 3: Advanced Evaluation Methods
```bash
python part3_advanced_eval.py
```

**What it does:**
- Compares two evaluation approaches:
  - **Few-shot learning**: Uses example evaluations to guide the judge
  - **Rubric-based**: Uses explicit scoring criteria
- Outputs to `../output/advanced_evaluation_results.json`
- Prints comparison statistics

**Prerequisites:** Must run Part 1 first

**Runtime:** ~30-50 minutes

---

#### Part 3 (Alternative): Multi-Agent Debate
```bash
python part3_multi_agent_debate.py
```

**What it does:**
- Simulates multi-agent debates for mediation evaluation
- Implements debate rounds between multiple judge agents
- Outputs to `../output/user_sim_results_full.json`

**Prerequisites:** Must run Part 1 first

**Runtime:** ~40-60 minutes (longer due to multiple debate rounds)

---

#### Part 5: Comparative Analysis
```bash
python part5_comparative_analysis.py
```

**What it does:**
- Performs comprehensive comparative analysis across all methods
- Generates statistical comparisons and visualizations
- Outputs to `../output/combined_results.json`

**Prerequisites:** Must run Parts 1-3 first

**Runtime:** ~20-30 minutes

## Configuration Options

### Modifying Model Selection

Edit the `MODEL_NAME` or `JUDGE_MODEL_NAME` variables in each script:

```python
# In part1_mediation.py
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

# In part2_evaluation.py, part3_advanced_eval.py
JUDGE_MODEL_NAME = "Qwen/Qwen3-4B"
```

### Adjusting Batch Size

If you encounter out-of-memory errors, reduce the batch size:

```python
BATCH_SIZE = 8  # Reduce to 4 or 2 if needed
```

### Modifying Output Paths

Change the output file paths at the top of each script:

```python
OUTPUT_FILE = "../output/mediation_outputs.json"
```

## Troubleshooting

### Out of Memory Errors
- Reduce `BATCH_SIZE` in the script
- Reduce `max_length` parameter in tokenizer calls
- Use a smaller model

### Model Download Issues
- Ensure you have Hugging Face access token configured
- Check internet connectivity
- Verify disk space for model cache (~20GB needed)

### Missing Dependencies
```bash
pip install --upgrade torch transformers accelerate huggingface-hub
```

## Output Files

All output files are saved in JSON format in the `output/` directory:

- **mediation_outputs.json**: Contains generated judgments and steering messages
- **evaluation_results.json**: Contains evaluation scores and rationales
- **advanced_evaluation_results.json**: Contains few-shot and rubric-based evaluations
- **user_sim_results_full.json**: Contains multi-agent debate results
- **combined_results.json**: Contains comprehensive comparative analysis

## Performance Notes

- **Batch Processing**: Scripts use batched inference for efficiency
- **Mixed Precision**: FP16 is used to reduce memory usage and increase speed
