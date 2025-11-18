#!/bin/bash
#SBATCH --job-name=tabpfn_synbeats_cpu
#SBATCH --output=tabpfn_synbeats_cpu_%j.out
#SBATCH --error=tabpfn_synbeats_cpu_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --partition=deho
#SBATCH --cpus-per-task=8

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Device Type: CPU"
echo "Feature Type: Synbeats-style"
echo "=========================================="

# Load any required modules (adjust based on your cluster)
# Example:
module load python/3.12.1

# Activate virtual environment if needed
source /scratch/users/salilg/envs/tabpfn_env/.venv/bin/activate

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
OUTPUT_FILE="tabpfn_predictions_synbeats_cpu_${TIMESTAMP}.csv"

# Run the prediction script with synbeats features
echo ""
echo "=========================================="
echo "Running TabPFNv2 predictions with synbeats features on CPU..."
echo "Output will be saved to: $OUTPUT_FILE"
echo "=========================================="
echo ""

python3 run_tabpfn_predictions_synbeats.py \
    --data-path ../smoking_data.csv \
    --output "$OUTPUT_FILE" \
    --treated-id 3 \
    --n-pcs 2

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
