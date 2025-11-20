#!/bin/bash
#SBATCH --job-name=tabpfn_panel_gpu
#SBATCH --output=tabpfn_panel_gpu_%j.out
#SBATCH --error=tabpfn_panel_gpu_%j.err
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
echo "Model: TabPFN with Panel Statistics"
echo "=========================================="

# Load any required modules (adjust based on your cluster)
# Example:
module load python/3.12.1
module load cuda

# Activate virtual environment if needed
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

# Print Python and GPU information
echo ""
echo "Python version:"
python3 --version

echo ""
echo "GPU information:"
nvidia-smi || echo "nvidia-smi not available"

# Set output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="tabpfn_predictions_panel_gpu_${TIMESTAMP}.csv"

# Run the prediction script
echo ""
echo "=========================================="
echo "Running TabPFN predictions with panel statistics on GPU..."
echo "Output will be saved to: $OUTPUT_FILE"
echo "Note: Running from $(pwd)"
echo "=========================================="
echo ""

python3 run_tabpfn_predictions_panel.py \
    --data-path ../smoking_data.csv \
    --output "$OUTPUT_FILE" \
    --treated-id 3 \
    --n-lags 5

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
