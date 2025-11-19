#!/bin/bash
#SBATCH --job-name=chronos2_smoking_gpu
#SBATCH --output=chronos2_smoking_gpu_%j.out
#SBATCH --error=chronos2_smoking_gpu_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=32GB
#SBATCH --partition=deho
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Device Type: GPU"
echo "Model: Chronos-2"
echo "=========================================="

# Load any required modules (adjust based on your cluster)
# Example:
module load python/3.12.1
module load cuda/11.8

# Activate virtual environment if needed
source /scratch/users/salilg/envs/chronos_env/.venv/bin/activate

# Print Python and GPU information
echo ""
echo "Python version:"
python3 --version

echo ""
echo "GPU information:"
nvidia-smi || echo "nvidia-smi not available"

# Set Hugging Face cache directory to scratch (models are ~1-2GB)
export HF_HOME=/scratch/users/salilg/huggingface_cache
echo "HF_HOME set to: $HF_HOME"

# Set output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run the prediction script
echo ""
echo "=========================================="
echo "Running Chronos-2 predictions on GPU..."
echo "Note: Running from $(pwd)"
echo "Models will be cached in: $HF_HOME"
echo "=========================================="
echo ""

# Run Variant B (recommended) by default
# Change --variant to 'all' to run all variants
python3 run_chronos2_predictions.py \
    --data-path ../smoking_data.csv \
    --output-dir . \
    --treated-id 3 \
    --variant B

# Check if script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job completed successfully!"
    echo "End Time: $(date)"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Job failed with error code $?"
    echo "End Time: $(date)"
    echo "=========================================="
    exit 1
fi
