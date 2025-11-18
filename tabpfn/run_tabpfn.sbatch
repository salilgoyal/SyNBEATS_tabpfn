#!/bin/bash
#SBATCH --job-name=tabpfn_smoking
#SBATCH --output=tabpfn_smoking_%j.out
#SBATCH --error=tabpfn_smoking_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32GB

# ============================================================================
# CONFIGURATION: Set device type
# ============================================================================
# Options: "cpu" or "gpu"
DEVICE_TYPE="gpu"
# ============================================================================

# GPU-specific SLURM directives (only used if DEVICE_TYPE=gpu)
if [ "$DEVICE_TYPE" == "gpu" ]; then
    #SBATCH --partition=gpu
    #SBATCH --gres=gpu:1
    #SBATCH --cpus-per-task=4
    echo "Requesting GPU resources"
else
    #SBATCH --partition=normal
    #SBATCH --cpus-per-task=8
    echo "Requesting CPU resources"
fi

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Device Type: $DEVICE_TYPE"
echo "=========================================="

# Load any required modules (adjust based on your cluster)
# Example:
# module load python/3.9
# module load cuda/11.8  # if using GPU

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Print Python and GPU information
echo ""
echo "Python version:"
python3 --version

if [ "$DEVICE_TYPE" == "gpu" ]; then
    echo ""
    echo "GPU information:"
    nvidia-smi || echo "nvidia-smi not available"
fi

# Navigate to the script directory
cd tabpfn/

# Set output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="tabpfn_predictions_${TIMESTAMP}.csv"

# Run the prediction script
echo ""
echo "=========================================="
echo "Running TabPFNv2 predictions..."
echo "Output will be saved to: $OUTPUT_FILE"
echo "=========================================="
echo ""

python3 run_tabpfn_predictions.py \
    --data-path ../smoking_data.csv \
    --output "$OUTPUT_FILE" \
    --treated-id 3

# Check if script completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Job completed successfully!"
    echo "Output file: $OUTPUT_FILE"
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
