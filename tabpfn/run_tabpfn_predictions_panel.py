#!/usr/bin/env python3
"""
TabPFNv2 predictions using panel statistics features.
Uses the same panel statistics as Chronos-2 Variant B for direct comparison.

Features include: panel mean, std, median, quantiles (q10, q25, q75, q90),
skewness, kurtosis, time indices, and target state's own lags.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
from tabpfn.regressor import TabPFNRegressor
from tabpfn.constants import ModelVersion

# ============================================================================
# CONFIGURATION: Set device preference
# ============================================================================
USE_GPU = True  # Set to False to use CPU only
# ============================================================================

def setup_device():
    """Configure device based on USE_GPU setting."""
    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print("Using device: CPU")
    else:
        print("Using device: GPU (if available)")

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
    Compute panel-level statistics excluding specified states.

    Same function as in Chronos-2 script for consistency.

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
        ('panel_sd', 'std'),
        ('panel_median', 'median'),
        ('panel_q25', lambda x: x.quantile(0.25)),
        ('panel_q75', lambda x: x.quantile(0.75)),
        ('panel_q10', lambda x: x.quantile(0.1)),
        ('panel_q90', lambda x: x.quantile(0.9)),
        ('panel_skewness', lambda x: x.skew()),
        ('panel_kurtosis', lambda x: x.kurtosis()),
    ]).reset_index()
    return stats

def build_features_for_state(dta, state_id, treated_id, pre_start, pre_end,
                              post_start, post_end, n_lags=5):
    """
    Build feature matrix for a single state using panel statistics.

    Parameters:
    -----------
    dta : pd.DataFrame
        Full dataset
    state_id : int
        State to build features for
    treated_id : int
        Treated state to exclude
    pre_start, pre_end : int
        Training period boundaries
    post_start, post_end : int
        Prediction period boundaries
    n_lags : int
        Number of past lags to include for target state

    Returns:
    --------
    X_train, y_train, X_test : np.ndarray
        Training and test feature matrices
    """
    # Compute panel statistics excluding both treated and target state
    exclude_list = [treated_id, state_id]
    panel_stats = compute_panel_statistics(dta, exclude_list)

    # Get state's own data
    state_data = dta[dta['id'] == state_id].sort_values('time')
    state_dict = dict(zip(state_data['time'], state_data['Y_obs']))

    # Time feature base
    t0 = pre_start

    # Build training features
    X_train, y_train = [], []
    for year in range(pre_start, pre_end + 1):
        if year not in state_dict:
            continue

        y_train.append(state_dict[year])

        # Get panel statistics for this year
        panel_year = panel_stats[panel_stats['time'] == year]
        if len(panel_year) == 0:
            continue

        # Panel statistics features
        feat = [
            panel_year['panel_mean'].iloc[0],
            panel_year['panel_sd'].iloc[0],
            panel_year['panel_median'].iloc[0],
            panel_year['panel_q25'].iloc[0],
            panel_year['panel_q75'].iloc[0],
            panel_year['panel_q10'].iloc[0],
            panel_year['panel_q90'].iloc[0],
            panel_year['panel_skewness'].iloc[0],
            panel_year['panel_kurtosis'].iloc[0],
        ]

        # Time features
        t = year - t0
        feat.extend([float(t), float(t ** 2)])

        # Target state's own lags
        for lag in range(1, n_lags + 1):
            feat.append(state_dict.get(year - lag, np.nan))

        X_train.append(feat)

    # Build test features
    X_test = []
    for year in range(post_start, post_end + 1):
        # Get panel statistics for this year
        panel_year = panel_stats[panel_stats['time'] == year]
        if len(panel_year) == 0:
            continue

        # Panel statistics features (same as training)
        feat = [
            panel_year['panel_mean'].iloc[0],
            panel_year['panel_sd'].iloc[0],
            panel_year['panel_median'].iloc[0],
            panel_year['panel_q25'].iloc[0],
            panel_year['panel_q75'].iloc[0],
            panel_year['panel_q10'].iloc[0],
            panel_year['panel_q90'].iloc[0],
            panel_year['panel_skewness'].iloc[0],
            panel_year['panel_kurtosis'].iloc[0],
        ]

        # Time features
        t = year - t0
        feat.extend([float(t), float(t ** 2)])

        # Target state's own lags (using actual observations, not predictions)
        for lag in range(1, n_lags + 1):
            feat.append(state_dict.get(year - lag, np.nan))

        X_test.append(feat)

    return np.array(X_train), np.array(y_train), np.array(X_test)

def run_tabpfn_predictions_panel(dta, treated_id=3, n_lags=5):
    """
    Run TabPFNv2 predictions using panel statistics features.

    Parameters:
    -----------
    dta : pd.DataFrame
        Input data with columns ['id', 'time', 'Y_obs']
    treated_id : int
        ID of the treated state to exclude (default: 3 for California)
    n_lags : int
        Number of past lags to include

    Returns:
    --------
    dict
        Dictionary mapping (state_id, year) to predicted values
    """
    # Basic metadata
    state_ids = sorted(int(s) for s in dta['id'].unique())
    pre_start, pre_end = 1970, 1983
    post_start, post_end = 1984, 2000

    print(f"\nTraining on years {pre_start}-{pre_end}")
    print(f"Predicting years {post_start}-{post_end}")
    print(f"Number of states: {len(state_ids)}")
    print(f"Excluding treated state (id={treated_id})")
    print(f"Using panel statistics with {n_lags} target lags\n")

    # Dictionary to store predictions
    predictions_dict = {}

    for state_id in tqdm(state_ids, desc="Processing states"):
        if state_id == treated_id:
            continue  # skip treated state

        # Build features for this state
        X_train, y_train, X_test = build_features_for_state(
            dta, state_id, treated_id,
            pre_start, pre_end, post_start, post_end,
            n_lags=n_lags
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Warning: No data for state {state_id}, skipping")
            continue

        # Fit TabPFNv2 model
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Store predictions
        for i, year in enumerate(range(post_start, post_end + 1)):
            if i < len(y_pred):
                predictions_dict[(state_id, year)] = float(y_pred[i])

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
        description='Run TabPFNv2 predictions with panel statistics features'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='../smoking_data.csv',
        help='Path to smoking_data.csv (default: ../smoking_data.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='tabpfn_predictions_panel.csv',
        help='Output CSV filename (default: tabpfn_predictions_panel.csv)'
    )
    parser.add_argument(
        '--treated-id',
        type=int,
        default=3,
        help='ID of treated state to exclude (default: 3 for California)'
    )
    parser.add_argument(
        '--n-lags',
        type=int,
        default=5,
        help='Number of past lags to include (default: 5)'
    )

    args = parser.parse_args()

    # Setup device
    setup_device()

    # Load data
    print(f"Loading data from: {args.data_path}")
    dta = load_smoking_data(args.data_path)
    print(f"Loaded {len(dta)} observations")

    # Run predictions
    predictions_dict = run_tabpfn_predictions_panel(
        dta,
        treated_id=args.treated_id,
        n_lags=args.n_lags
    )

    # Create output DataFrame
    out = create_output_dataframe(predictions_dict, dta, treated_id=args.treated_id)

    # Save to CSV
    output_path = args.output
    out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Output shape: {out.shape}")

    # Print sample results
    print("\nSample predictions for year 1989:")
    sample = out[out['year'] == 1989].head(10)
    if len(sample) > 0:
        print(sample)

    # Print summary statistics
    post_period = out[out['year'] >= 1984]
    if len(post_period) > 0:
        print(f"\nPost-period summary (1984-2000):")
        print(f"Mean absolute gap: {post_period['gap'].abs().mean():.2f}")
        print(f"RMSE: {np.sqrt((post_period['gap'] ** 2).mean()):.2f}")

if __name__ == "__main__":
    main()
