#!/bin/bash
#SBATCH --job-name=tabpfn_smoking_cpu
#SBATCH --output=tabpfn_smoking_cpu_%j.out
#SBATCH --error=tabpfn_smoking_cpu_%j.err
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
echo "=========================================="

# Load any required modules (adjust based on your cluster)
# Example:
# module load python/3.9

# Activate virtual environment if needed
# source /path/to/your/venv/bin/activate

# Print Python information
echo ""
echo "Python version:"
python3 --version

echo ""
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

# Navigate to the script directory
cd tabpfn/

# Force CPU usage by setting environment variable
export CUDA_VISIBLE_DEVICES=""

# Set output filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_FILE="tabpfn_predictions_cpu_${TIMESTAMP}.csv"

# Run the prediction script
echo ""
echo "=========================================="
echo "Running TabPFNv2 predictions on CPU..."
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
