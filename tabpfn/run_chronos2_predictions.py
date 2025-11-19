#!/usr/bin/env python3
"""
Chronos-2 predictions for smoking data (excluding California, id=3).
Generates predictions for each control state using time series foundation model.

Implements three variants:
- Variant A: Pure univariate (baseline)
- Variant B: With panel covariates (recommended)
- Variant C: Cross-learning with joint prediction
"""

import os
import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION: Set device preference
# ============================================================================
USE_GPU = True  # Set to False to use CPU only
# ============================================================================

def setup_device():
    """Configure device based on USE_GPU setting."""
    if USE_GPU:
        device = "cuda"
        print("Using device: GPU (CUDA)")
    else:
        device = "cpu"
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Using device: CPU")
    return device

def load_smoking_data(data_path):
    """
    Load and preprocess smoking data.

    Parameters:
    -----------
    data_path : str
        Path to the smoking_data.csv file

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['id', 'time', 'Y_obs']
    """
    dta = pd.read_csv(data_path)
    dta = dta[["state", "year", "cigsale"]]
    dta.columns = ["id", "time", "Y_obs"]
    return dta

def compute_panel_statistics(dta, exclude_states):
    """
    Compute panel-level statistics (mean, std) excluding specified states.

    Parameters:
    -----------
    dta : pd.DataFrame
        Input data with columns ['id', 'time', 'Y_obs']
    exclude_states : list
        List of state IDs to exclude

    Returns:
    --------
    pd.DataFrame
        DataFrame with panel statistics per time period
    """
    panel_data = dta[~dta['id'].isin(exclude_states)]
    stats = panel_data.groupby('time')['Y_obs'].agg([
        ('panel_mean', 'mean'),
        ('panel_sd', 'std')
    ]).reset_index()
    return stats

def prepare_chronos_data_variant_a(dta, state_id, pre_start, pre_end):
    """
    Variant A: Pure univariate - only use the state's own history.

    Parameters:
    -----------
    dta : pd.DataFrame
        Full dataset
    state_id : int
        State to prepare data for
    pre_start, pre_end : int
        Training period boundaries

    Returns:
    --------
    dict
        Input dictionary for Chronos-2
    """
    state_data = dta[dta['id'] == state_id].sort_values('time')
    train_data = state_data[(state_data['time'] >= pre_start) &
                            (state_data['time'] <= pre_end)]

    return {
        "target": train_data['Y_obs'].values
    }

def prepare_chronos_data_variant_b(dta, state_id, treated_id,
                                   pre_start, pre_end, post_start, post_end):
    """
    Variant B: With panel covariates (recommended).
    Uses panel statistics as covariates.

    Parameters:
    -----------
    dta : pd.DataFrame
        Full dataset
    state_id : int
        State to prepare data for
    treated_id : int
        Treated state to exclude
    pre_start, pre_end : int
        Training period boundaries
    post_start, post_end : int
        Prediction period boundaries

    Returns:
    --------
    dict
        Input dictionary for Chronos-2 with covariates
    """
    # Get state's own data
    state_data = dta[dta['id'] == state_id].sort_values('time')
    train_data = state_data[(state_data['time'] >= pre_start) &
                            (state_data['time'] <= pre_end)]

    # Compute panel statistics excluding both treated and target state
    exclude_list = [treated_id, state_id]
    panel_stats = compute_panel_statistics(dta, exclude_list)

    # Get training period panel stats
    train_panel = panel_stats[(panel_stats['time'] >= pre_start) &
                              (panel_stats['time'] <= pre_end)]

    # Get prediction period panel stats (known from other states)
    future_panel = panel_stats[(panel_stats['time'] >= post_start) &
                               (panel_stats['time'] <= post_end)]

    # Time features
    t0 = pre_start
    train_times = train_data['time'].values
    future_times = np.arange(post_start, post_end + 1)

    train_t = train_times - t0
    train_t2 = train_t ** 2
    future_t = future_times - t0
    future_t2 = future_t ** 2

    return {
        "target": train_data['Y_obs'].values,
        "past_covariates": {
            "panel_mean": train_panel['panel_mean'].values,
            "panel_sd": train_panel['panel_sd'].values,
            "time_index": train_t.astype(float),
            "time_index_sq": train_t2.astype(float),
        },
        "future_covariates": {
            "panel_mean": future_panel['panel_mean'].values,
            "panel_sd": future_panel['panel_sd'].values,
            "time_index": future_t.astype(float),
            "time_index_sq": future_t2.astype(float),
        }
    }

def run_chronos2_variant_a(dta, pipeline, treated_id=3):
    """
    Variant A: Pure univariate forecasting.

    Returns:
    --------
    dict
        Dictionary mapping (state_id, year) to predicted values
    """
    state_ids = sorted(int(s) for s in dta['id'].unique())
    pre_start, pre_end = 1970, 1983
    post_start, post_end = 1984, 2000
    prediction_length = post_end - post_start + 1

    print("\n=== VARIANT A: Pure Univariate ===")
    print(f"Training on years {pre_start}-{pre_end}")
    print(f"Predicting years {post_start}-{post_end}")

    predictions_dict = {}

    for state_id in tqdm(state_ids, desc="Variant A"):
        if state_id == treated_id:
            continue

        # Prepare input
        input_dict = prepare_chronos_data_variant_a(
            dta, state_id, pre_start, pre_end
        )

        # Make predictions
        quantiles, mean = pipeline.predict_quantiles(
            [input_dict],
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9]
        )

        # Extract median predictions
        preds = mean[0].cpu().numpy().flatten()

        # Store predictions
        for i, year in enumerate(range(post_start, post_end + 1)):
            predictions_dict[(state_id, year)] = float(preds[i])

    return predictions_dict

def run_chronos2_variant_b(dta, pipeline, treated_id=3):
    """
    Variant B: With panel covariates (recommended).

    Returns:
    --------
    dict
        Dictionary mapping (state_id, year) to predicted values
    """
    state_ids = sorted(int(s) for s in dta['id'].unique())
    pre_start, pre_end = 1970, 1983
    post_start, post_end = 1984, 2000
    prediction_length = post_end - post_start + 1

    print("\n=== VARIANT B: With Panel Covariates ===")
    print(f"Training on years {pre_start}-{pre_end}")
    print(f"Predicting years {post_start}-{post_end}")

    predictions_dict = {}

    for state_id in tqdm(state_ids, desc="Variant B"):
        if state_id == treated_id:
            continue

        # Prepare input with covariates
        input_dict = prepare_chronos_data_variant_b(
            dta, state_id, treated_id,
            pre_start, pre_end, post_start, post_end
        )

        # Make predictions
        quantiles, mean = pipeline.predict_quantiles(
            [input_dict],
            prediction_length=prediction_length,
            quantile_levels=[0.1, 0.5, 0.9]
        )

        # Extract median predictions
        preds = mean[0].cpu().numpy().flatten()

        # Store predictions
        for i, year in enumerate(range(post_start, post_end + 1)):
            predictions_dict[(state_id, year)] = float(preds[i])

    return predictions_dict

def run_chronos2_variant_c(dta, pipeline, treated_id=3):
    """
    Variant C: Cross-learning with joint prediction.

    Returns:
    --------
    dict
        Dictionary mapping (state_id, year) to predicted values
    """
    state_ids = sorted(int(s) for s in dta['id'].unique())
    control_states = [s for s in state_ids if s != treated_id]
    pre_start, pre_end = 1970, 1983
    post_start, post_end = 1984, 2000
    prediction_length = post_end - post_start + 1

    print("\n=== VARIANT C: Cross-Learning (Joint Prediction) ===")
    print(f"Training on years {pre_start}-{pre_end}")
    print(f"Predicting years {post_start}-{post_end}")
    print(f"Using {len(control_states)} control states jointly")

    predictions_dict = {}

    # Prepare data for all control states
    print("Preparing input data for all control states...")
    all_inputs = []
    for state_id in control_states:
        input_dict = prepare_chronos_data_variant_a(
            dta, state_id, pre_start, pre_end
        )
        all_inputs.append(input_dict)

    # Make joint predictions for all states
    print("Making joint predictions...")
    quantiles, mean = pipeline.predict_quantiles(
        all_inputs,
        prediction_length=prediction_length,
        quantile_levels=[0.1, 0.5, 0.9],
        # Note: predict_batches_jointly not available in predict_quantiles
        # This variant uses the batch context implicitly
    )

    # Store predictions
    for idx, state_id in enumerate(tqdm(control_states, desc="Storing predictions")):
        preds = mean[idx].cpu().numpy().flatten()
        for i, year in enumerate(range(post_start, post_end + 1)):
            predictions_dict[(state_id, year)] = float(preds[i])

    return predictions_dict

def create_output_dataframe(predictions_dict, dta, treated_id=3):
    """
    Create final output DataFrame with predictions and gaps.

    Parameters:
    -----------
    predictions_dict : dict
        Dictionary with predictions
    dta : pd.DataFrame
        Original data
    treated_id : int
        ID of treated state to exclude

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['id', 'year', 'predicted', 'obs', 'gap']
    """
    # Create observed data dict
    dta_dict = {(int(row['id']), int(row['time'])): float(row['Y_obs'])
                for _, row in dta.iterrows()}

    pred_df = pd.DataFrame(
        [(i, yr, val) for (i, yr), val in predictions_dict.items()],
        columns=["id", "year", "predicted"]
    )

    obs_df = pd.DataFrame(
        [(i, yr, val) for (i, yr), val in dta_dict.items() if i != treated_id],
        columns=["id", "year", "obs"]
    )

    out = obs_df.merge(pred_df, on=["id", "year"], how="left")
    out["gap"] = out["obs"] - out["predicted"]

    out = out.sort_values(["id", "year"]).reset_index(drop=True)
    out = out[["id", "year", "predicted", "obs", "gap"]]

    return out

def main():
    parser = argparse.ArgumentParser(
        description='Run Chronos-2 predictions on smoking data'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../smoking_data.csv',
        help='Path to smoking_data.csv (default: ../smoking_data.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for CSV files (default: current directory)'
    )
    parser.add_argument(
        '--treated-id',
        type=int,
        default=3,
        help='ID of treated state to exclude (default: 3 for California)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        choices=['A', 'B', 'C', 'all'],
        default='B',
        help='Variant to run: A (univariate), B (covariates), C (cross-learning), all (default: B)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='amazon/chronos-2',
        help='Chronos model to use (default: amazon/chronos-2)'
    )

    args = parser.parse_args()

    # Setup device
    device = setup_device()

    # Load Chronos-2 pipeline
    print(f"\nLoading Chronos-2 model: {args.model}")
    print("This may take a few minutes on first run...")

    try:
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(
            args.model,
            device_map=device
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure chronos-forecasting is installed:")
        print("  pip install 'chronos-forecasting>=2.0'")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from: {args.data_path}")
    dta = load_smoking_data(args.data_path)
    print(f"Loaded {len(dta)} observations")

    # Original data dict for comparison
    dta_dict = {(int(row['id']), int(row['time'])): float(row['Y_obs'])
                for _, row in dta.iterrows()}

    # Run selected variant(s)
    variants_to_run = ['A', 'B', 'C'] if args.variant == 'all' else [args.variant]

    for variant in variants_to_run:
        print(f"\n{'='*60}")
        print(f"Running Variant {variant}")
        print(f"{'='*60}")

        if variant == 'A':
            predictions_dict = run_chronos2_variant_a(dta, pipeline, args.treated_id)
            output_file = f"{args.output_dir}/chronos2_predictions_variant_a.csv"
        elif variant == 'B':
            predictions_dict = run_chronos2_variant_b(dta, pipeline, args.treated_id)
            output_file = f"{args.output_dir}/chronos2_predictions_variant_b.csv"
        elif variant == 'C':
            predictions_dict = run_chronos2_variant_c(dta, pipeline, args.treated_id)
            output_file = f"{args.output_dir}/chronos2_predictions_variant_c.csv"

        # Create output DataFrame
        out = create_output_dataframe(predictions_dict, dta, treated_id=args.treated_id)

        # Save to CSV
        out.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Output shape: {out.shape}")

        # Print sample results
        print(f"\nSample predictions for year 1989:")
        sample = out[out['year'] == 1989].head(10)
        if len(sample) > 0:
            print(sample)

        # Print summary statistics
        post_period = out[out['year'] >= 1984]
        if len(post_period) > 0:
            print(f"\nPost-period summary (1984-2000):")
            print(f"Mean absolute gap: {post_period['gap'].abs().mean():.2f}")
            print(f"RMSE: {np.sqrt((post_period['gap'] ** 2).mean()):.2f}")

    print(f"\n{'='*60}")
    print("All variants completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
