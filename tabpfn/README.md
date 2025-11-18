# TabPFNv2 Smoking Data Predictions

This directory contains scripts to run TabPFNv2 predictions on the smoking dataset, excluding California (id=3) and generating predictions for all control states.

## Two Feature Engineering Approaches

We provide two different feature engineering approaches:

1. **Basic Window Features** (`run_tabpfn_predictions.py`)
   - Uses raw cigarette sales from other states in ±5 year windows
   - Simple and fast
   - Direct recursive forecasting

2. **Synbeats-Style Features** (`run_tabpfn_predictions_synbeats.py`)
   - Enhanced feature engineering with:
     - Time indices (t, t²)
     - Within-state features (lags, differences, rolling means)
     - Panel statistics (cross-state means and standard deviations)
     - PCA factors across states
     - State-specific linear trends and residuals
   - More sophisticated but potentially more accurate

## Files

### Basic Window Features
- `run_tabpfn_predictions.py` - Python script with basic window features
- `run_tabpfn_gpu.sbatch` - SLURM script for GPU execution
- `run_tabpfn_cpu.sbatch` - SLURM script for CPU execution
- `run_tabpfn.sbatch` - Configurable SLURM script (CPU/GPU)

### Synbeats-Style Features
- `run_tabpfn_predictions_synbeats.py` - Python script with synbeats features
- `run_tabpfn_synbeats_gpu.sbatch` - SLURM script for GPU execution
- `run_tabpfn_synbeats_cpu.sbatch` - SLURM script for CPU execution

### Other
- `tabpfn.ipynb` - Original notebook with exploratory analysis
- `README.md` - This file

## Quick Start

### Basic Window Features

**Run locally:**
```bash
cd tabpfn/
python3 run_tabpfn_predictions.py
```

**Submit to SLURM (GPU):**
```bash
sbatch tabpfn/run_tabpfn_gpu.sbatch
```

**Submit to SLURM (CPU):**
```bash
sbatch tabpfn/run_tabpfn_cpu.sbatch
```

### Synbeats-Style Features

**Run locally:**
```bash
cd tabpfn/
python3 run_tabpfn_predictions_synbeats.py
```

**Submit to SLURM (GPU - recommended):**
```bash
sbatch tabpfn/run_tabpfn_synbeats_gpu.sbatch
```

**Submit to SLURM (CPU):**
```bash
sbatch tabpfn/run_tabpfn_synbeats_cpu.sbatch
```

## Configuration

### CPU vs GPU

**Easy Method (Recommended):** Use the dedicated sbatch files:
- For GPU: `sbatch tabpfn/run_tabpfn_gpu.sbatch`
- For CPU: `sbatch tabpfn/run_tabpfn_cpu.sbatch`

**Manual Method:** Edit configuration in files:

1. **In the Python script** (`run_tabpfn_predictions.py`):
   - Edit line 20: `USE_GPU = True` or `USE_GPU = False`

2. **In the configurable sbatch file** (`run_tabpfn.sbatch`):
   - Edit line 11: `DEVICE_TYPE="gpu"` or `DEVICE_TYPE="cpu"`

**Note:** The dedicated `run_tabpfn_gpu.sbatch` and `run_tabpfn_cpu.sbatch` files automatically set the correct configuration.

### Command-line Arguments

**Basic Window Features:**
```bash
python3 run_tabpfn_predictions.py \
    --data-path ../smoking_data.csv \
    --output tabpfn_predictions.csv \
    --treated-id 3
```

**Synbeats-Style Features:**
```bash
python3 run_tabpfn_predictions_synbeats.py \
    --data-path ../smoking_data.csv \
    --output tabpfn_predictions_synbeats.csv \
    --treated-id 3 \
    --n-pcs 2
```

**Common arguments:**
- `--data-path`: Path to the smoking data CSV (default: `../smoking_data.csv`)
- `--output`: Output CSV filename (defaults differ by script)
- `--treated-id`: ID of the treated state to exclude (default: `3` for California)

**Synbeats-specific arguments:**
- `--n-pcs`: Number of PCA components to compute (default: `2`)

## Methods

### Basic Window Features Method

1. **Data Preparation**: Loads smoking data and creates a dictionary mapping (state_id, year) to cigarette sales
2. **Training**: For each control state (excluding California):
   - Trains on pre-treatment period (1970-1983)
   - Uses raw cigarette sales from other control states with:
     - 5 past lags (w=5)
     - 5 future lookahead values (r=5)
   - Uses TabPFNv2 regressor
3. **Prediction**: Recursively predicts post-treatment period (1984-2000)
4. **Output**: Saves predictions with observed values and gaps to CSV

### Synbeats-Style Features Method

1. **Feature Engineering**: Builds rich feature set for all states:
   - Time indices (t, t²)
   - Within-state lags (1-3 periods), differences, rolling means
   - Panel statistics (means, standard deviations across states)
   - PCA factors capturing common trends
   - State-specific linear trends and residuals
2. **Training**: For each control state:
   - Excludes both California and the target state from panel statistics
   - Trains on pre-treatment period (1970-1983)
   - Uses enhanced features in ±5 year windows
   - Uses TabPFNv2 regressor
3. **Prediction**: Predicts post-treatment period (1984-2000) using control state features
4. **Output**: Saves predictions with observed values and gaps to CSV

## Output Format

The output CSV contains the following columns:

- `id`: State identifier
- `year`: Year
- `predicted`: TabPFNv2 prediction
- `obs`: Observed value
- `gap`: Observed - Predicted

## SLURM Configuration

The sbatch file is configured for:

- **Time limit**: 4 hours
- **Memory**: 32GB
- **GPU mode**: 1 GPU, 4 CPUs
- **CPU mode**: 8 CPUs

Adjust these settings in `run_tabpfn.sbatch` based on your cluster specifications.

## Dependencies

**Required for all scripts:**
- pandas
- numpy
- tqdm
- tabpfn

**Additional requirement for synbeats features:**
- scikit-learn (for PCA)

**Install all dependencies:**
```bash
pip install pandas numpy tqdm tabpfn scikit-learn
```

**Or use requirements.txt:**
```bash
pip install -r ../requirements.txt
```

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View output:
```bash
tail -f tabpfn_smoking_<job_id>.out
```

Cancel job:
```bash
scancel <job_id>
```
