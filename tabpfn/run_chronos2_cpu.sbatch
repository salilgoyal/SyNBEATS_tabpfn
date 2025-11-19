#!/bin/bash
#SBATCH --job-name=chronos2_smoking_cpu
#SBATCH --output=chronos2_smoking_cpu_%j.out
#SBATCH --error=chronos2_smoking_cpu_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Device Type: CPU"
echo "Model: Chronos-2"
echo "=========================================="

# Load any required modules (adjust based on your cluster)
# Example:
# module load python/3.12.1

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Print Python information
echo ""
echo "Python version:"
python3 --version

echo ""
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Force CPU usage by setting environment variable
export CUDA_VISIBLE_DEVICES=""

# Set Hugging Face cache directory to scratch (models are ~1-2GB)
export HF_HOME=/scratch/users/salilg/huggingface_cache
echo "HF_HOME set to: $HF_HOME"

# Set output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Run the prediction script
echo ""
echo "=========================================="
echo "Running Chronos-2 predictions on CPU..."
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
