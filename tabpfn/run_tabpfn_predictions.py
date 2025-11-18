#!/usr/bin/env python3
"""
TabPFNv2 predictions for smoking data (excluding California, id=3).
Generates predictions for each control state using recursive forecasting.
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
        # Force CPU usage
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

def run_tabpfn_predictions(dta, treated_id=3):
    """
    Run TabPFNv2 predictions for all control states (excluding treated unit).

    Parameters:
    -----------
    dta : pd.DataFrame
        Input data with columns ['id', 'time', 'Y_obs']
    treated_id : int
        ID of the treated state to exclude (default: 3 for California)

    Returns:
    --------
    dict
        Dictionary mapping (state_id, year) to predicted values
    """
    # Build the dict once (robust to dtype)
    dta_dict = {(int(row['id']), int(row['time'])): float(row['Y_obs'])
                for _, row in dta.iterrows()}

    # Will hold recursive predictions
    y_test = copy.deepcopy(dta_dict)

    # Basic metadata
    state_ids = sorted(int(s) for s in dta['id'].unique())
    pre_start, pre_end = 1970, 1983   # train on pre-treatment period
    post_start, post_end = 1984, 2000  # predict on post-treatment period
    w = 5  # past lags of the target
    r = 5  # future lookahead for controls

    print(f"\nTraining on years {pre_start}-{pre_end}")
    print(f"Predicting years {post_start}-{post_end}")
    print(f"Number of states: {len(state_ids)}")
    print(f"Excluding treated state (id={treated_id})")
    print(f"Past lags (w): {w}, Future lookahead (r): {r}\n")

    for i in tqdm(state_ids, desc="Processing states"):
        if i == treated_id:
            continue  # skip treated state (California)

        # --------- Build train set ----------
        X_train, y_train = [], []
        for year in range(pre_start, pre_end + 1):
            y_train.append(dta_dict[(i, year)])
            feat = []

            # Features from all other control states (excluding treated and current)
            for j in state_ids:
                if j in (treated_id, i):
                    continue
                for y in range(year - w, year + r + 1):
                    feat.append(dta_dict.get((j, y), np.nan))

            # Past lags of current state
            for y in range(year - w, year):
                feat.append(dta_dict.get((i, y), np.nan))

            X_train.append(feat)

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        # Fit TabPFNv2 model
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        model.fit(X_train, y_train)

        # --------- Recursive post-period prediction ----------
        for year in range(post_start, post_end + 1):
            feat = []

            # Features from all other control states (excluding treated and current)
            for j in state_ids:
                if j in (treated_id, i):
                    continue
                for y in range(year - w, year + r + 1):
                    feat.append(dta_dict.get((j, y), np.nan))

            # Past lags of current state (using recursive predictions)
            for y in range(year - w, year):
                feat.append(y_test.get((i, y), np.nan))

            yhat = float(model.predict(np.asarray([feat], dtype=float))[0])
            y_test[(i, year)] = yhat

    return y_test, dta_dict

def create_output_dataframe(y_test, dta_dict, treated_id=3):
    """
    Create final output DataFrame with predictions and gaps.

    Parameters:
    -----------
    y_test : dict
        Dictionary with predictions
    dta_dict : dict
        Dictionary with observed values
    treated_id : int
        ID of treated state to exclude

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['id', 'year', 'predicted', 'obs', 'gap']
    """
    pred_df = pd.DataFrame(
        [(i, yr, val) for (i, yr), val in y_test.items() if i != treated_id],
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
        description='Run TabPFNv2 predictions on smoking data'
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
        default='tabpfn_predictions.csv',
        help='Output CSV filename (default: tabpfn_predictions.csv)'
    )
    parser.add_argument(
        '--treated-id',
        type=int,
        default=3,
        help='ID of treated state to exclude (default: 3 for California)'
    )

    args = parser.parse_args()

    # Setup device
    setup_device()

    # Load data
    print(f"Loading data from: {args.data_path}")
    dta = load_smoking_data(args.data_path)
    print(f"Loaded {len(dta)} observations")

    # Run predictions
    y_test, dta_dict = run_tabpfn_predictions(dta, treated_id=args.treated_id)

    # Create output DataFrame
    out = create_output_dataframe(y_test, dta_dict, treated_id=args.treated_id)

    # Save to CSV
    output_path = args.output
    out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    main()
