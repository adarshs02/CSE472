#!/bin/bash
#SBATCH --job-name=cse472_part4           # Job name
#SBATCH --output=part4_%j.out             # Output file (%j = job ID)
#SBATCH --error=part4_%j.err              # Error file
#SBATCH --partition=gpu                   # GPU partition
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --cpus-per-task=8                 # Number of CPU cores
#SBATCH --mem=32G                         # Memory requirement
#SBATCH --time=24:00:00                   # Time limit (24 hours)
#SBATCH --mail-type=BEGIN,END,FAIL        # Email notifications
#SBATCH --mail-user=your_email@asu.edu    # Your email

# Print job information
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"
echo ""

# Load required modules
module purge
module load anaconda3
module load cuda/11.8

# Activate your conda environment (create it first if needed)
# conda create -n cse472 python=3.10
# conda activate cse472
# pip install torch transformers huggingface_hub accelerate

source activate cse472

# Print environment info
echo "Python version:"
python --version
echo ""

echo "CUDA available:"
python -c "import torch; print(torch.cuda.is_available())"
echo ""

echo "GPU info:"
nvidia-smi
echo ""

# Set Hugging Face token (set this as environment variable or pass as argument)
# export HF_TOKEN="your_huggingface_token_here"

# Define paths
DATASET_PATH="/path/to/your/dataset"  # UPDATE THIS PATH
OUTPUT_FILE="user_simulation_results_${SLURM_JOB_ID}.json"
SCRIPT_PATH="./cse472project2_part4_sol.py"

# Run the script with multiple threads
echo "Starting Part 4 simulation..."
echo "======================================"
python $SCRIPT_PATH \
    --dataset_path $DATASET_PATH \
    --output_file $OUTPUT_FILE \
    --num_threads 8 \
    --device cuda

# Print completion information
echo ""
echo "======================================"
echo "Job completed at: $(date)"
echo "Output saved to: $OUTPUT_FILE"
echo "======================================"
