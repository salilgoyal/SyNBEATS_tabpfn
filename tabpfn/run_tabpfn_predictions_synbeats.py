#!/usr/bin/env python3
"""
TabPFNv2 predictions for smoking data using synbeats-style features.
Generates predictions for each control state using enhanced feature engineering.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
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

def build_synbeats_style_features(dta, exclude_states=None, n_pcs=2):
    """
    Build enhanced features in synbeats style.

    Parameters:
    -----------
    dta : pd.DataFrame
        DataFrame with columns ['id', 'time', 'Y_obs']
    exclude_states : list or None
        List of state IDs to exclude from panel statistics and PCA
    n_pcs : int
        Number of PCA components (default: 2)

    Returns:
    --------
    feature_df : pd.DataFrame
        DataFrame with computed features
    feature_cols : list
        List of feature column names
    """
    if exclude_states is None:
        exclude_states = []

    df = dta.copy().sort_values(['id', 'time'])

    # Filter out excluded states for panel computations
    df_for_panel = df[~df['id'].isin(exclude_states)].copy()

    # ---------- 1. Time index ----------
    t0 = df['time'].min()
    df['t'] = df['time'] - t0
    df['t2'] = df['t'] ** 2

    # ---------- 2. Horizontal (within-state) features ----------
    g = df.groupby('id')

    df['lag1'] = g['Y_obs'].shift(1)
    df['lag2'] = g['Y_obs'].shift(2)
    df['lag3'] = g['Y_obs'].shift(3)

    # difference of lags (uses only past info)
    df['diff_lag'] = df['lag1'] - df['lag2']

    # mean of last 3 observations (t-1, t-2, t-3)
    df['mean_lag3'] = (df['lag1'] + df['lag2'] + df['lag3']) / 3.0

    # ---------- 3. Vertical (panel) features ----------
    # Panel stats per year (across control states only)
    time_stats = (
        df_for_panel.groupby('time')['Y_obs']
          .agg(panel_mean='mean', panel_sd='std')
          .reset_index()
    )
    df = df.merge(time_stats, on='time', how='left')

    # ---------- 4. PCA factors across states per year ----------
    # Pivot to (time x state) matrix, excluding treated states
    panel_wide = df_for_panel.pivot(index='time', columns='id', values='Y_obs').sort_index()

    # Simple interpolation for missing values
    panel_wide_interp = panel_wide.interpolate(limit_direction='both', axis=0)

    if len(panel_wide_interp.columns) >= n_pcs and len(panel_wide_interp) >= n_pcs:
        pca = PCA(n_components=n_pcs)
        pcs = pca.fit_transform(panel_wide_interp.values)
        pc_cols = [f'pc{k+1}' for k in range(n_pcs)]
        pc_df = (
            pd.DataFrame(pcs, index=panel_wide_interp.index, columns=pc_cols)
              .reset_index()
              .rename(columns={'index': 'time'})
        )
    else:
        # If not enough states/observations for PCA, create dummy columns
        pc_cols = [f'pc{k+1}' for k in range(n_pcs)]
        pc_df = pd.DataFrame({
            'time': df['time'].unique(),
            **{col: 0.0 for col in pc_cols}
        })

    df = df.merge(pc_df, on='time', how='left')

    # ---------- 5. State-specific linear trend & residual ----------
    trends = []
    for sid, g_state in df.groupby('id'):
        # need at least 2 points to fit a line
        valid_data = g_state[['t', 'Y_obs']].dropna()
        if len(valid_data) >= 2:
            slope, intercept = np.polyfit(valid_data['t'], valid_data['Y_obs'], 1)
        else:
            slope, intercept = np.nan, np.nan
        trends.append({'id': sid, 'trend_slope': slope, 'trend_intercept': intercept})

    trends = pd.DataFrame(trends)
    df = df.merge(trends, on='id', how='left')

    df['trend_pred'] = df['trend_intercept'] + df['trend_slope'] * df['t']
    df['residual'] = df['Y_obs'] - df['trend_pred']
    df['lag1_resid'] = df.groupby('id')['residual'].shift(1)

    # ---------- 6. Assemble final feature matrix ----------
    feature_cols = [
        'id',
        't', 't2',
        'lag1', 'lag2', 'lag3',
        'diff_lag', 'mean_lag3',
        'panel_mean', 'panel_sd',
    ] + pc_cols + [
        'trend_pred', 'lag1_resid'
    ]

    feature_df = df[['time', 'Y_obs'] + feature_cols].copy()

    return feature_df, feature_cols

def run_tabpfn_predictions_synbeats(dta, treated_id=3, n_pcs=2):
    """
    Run TabPFNv2 predictions using synbeats-style features.

    This version uses the synbeats features for the target state only,
    combined with raw values from control states in windows (similar to basic approach).

    Parameters:
    -----------
    dta : pd.DataFrame
        Input data with columns ['id', 'time', 'Y_obs']
    treated_id : int
        ID of the treated state to exclude (default: 3 for California)
    n_pcs : int
        Number of PCA components

    Returns:
    --------
    dict
        Dictionary mapping (state_id, year) to predicted values
    """
    # Basic metadata
    state_ids = sorted(int(s) for s in dta['id'].unique())
    pre_start, pre_end = 1970, 1983
    post_start, post_end = 1984, 2000
    w = 5  # window for training data
    r = 5  # future lookahead

    print(f"\nTraining on years {pre_start}-{pre_end}")
    print(f"Predicting years {post_start}-{post_end}")
    print(f"Number of states: {len(state_ids)}")
    print(f"Excluding treated state (id={treated_id})")
    print(f"Using synbeats-style features with {n_pcs} PCA components\n")

    # Dictionary to store predictions
    predictions_dict = {}

    # Original data dictionary for comparison
    dta_dict = {(int(row['id']), int(row['time'])): float(row['Y_obs'])
                for _, row in dta.iterrows()}

    for target_state in tqdm(state_ids, desc="Processing states"):
        if target_state == treated_id:
            continue  # skip treated state

        # Build features excluding both California and current target state
        exclude_list = [treated_id, target_state]
        feature_df, feature_cols = build_synbeats_style_features(
            dta,
            exclude_states=exclude_list,
            n_pcs=n_pcs
        )

        # --------- Build train set ----------
        X_train, y_train = [], []

        for year in range(pre_start, pre_end + 1):
            # Get target state's synbeats features for this year
            target_row = feature_df[
                (feature_df['id'] == target_state) &
                (feature_df['time'] == year)
            ]

            if len(target_row) == 0:
                continue

            y_train.append(float(target_row['Y_obs'].iloc[0]))

            # Extract synbeats features, replacing any NaN with 0
            target_features = []
            for col in feature_cols:
                val = target_row[col].iloc[0] if col in target_row.columns else np.nan
                target_features.append(0.0 if pd.isna(val) else float(val))

            target_features = np.array(target_features, dtype=float)

            # Add raw values from control states in window [year-w, year+r]
            control_raw = []
            for j in state_ids:
                if j in exclude_list:
                    continue
                for y in range(year - w, year + r + 1):
                    control_raw.append(dta_dict.get((j, y), np.nan))

            control_raw = np.array(control_raw, dtype=float)

            # Combine: synbeats features + control state raw values
            feat = np.concatenate([target_features, control_raw])
            X_train.append(feat)

        if len(X_train) == 0:
            print(f"Warning: No training data for state {target_state}, skipping")
            continue

        # Check that all feature vectors have the same length
        feat_lengths = [len(f) for f in X_train]
        if len(set(feat_lengths)) > 1:
            print(f"Error: Inconsistent feature lengths for state {target_state}: {set(feat_lengths)}")
            print(f"Feature lengths: {feat_lengths}")
            continue

        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        # Fit TabPFNv2 model
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2)
        model.fit(X_train, y_train)

        # --------- Predict post-period ----------
        for year in range(post_start, post_end + 1):
            # Get target state's synbeats features for prediction year
            target_row = feature_df[
                (feature_df['id'] == target_state) &
                (feature_df['time'] == year)
            ]

            if len(target_row) == 0:
                continue

            # Extract synbeats features, replacing any NaN with 0
            target_features = []
            for col in feature_cols:
                val = target_row[col].iloc[0] if col in target_row.columns else np.nan
                target_features.append(0.0 if pd.isna(val) else float(val))

            target_features = np.array(target_features, dtype=float)

            # Add raw values from control states in window
            control_raw = []
            for j in state_ids:
                if j in exclude_list:
                    continue
                for y in range(year - w, year + r + 1):
                    control_raw.append(dta_dict.get((j, y), np.nan))

            control_raw = np.array(control_raw, dtype=float)

            feat = np.concatenate([target_features, control_raw])

            # Ensure feature dimension matches training
            if len(feat) != X_train.shape[1]:
                print(f"Warning: Feature dimension mismatch for state {target_state}, year {year}")
                print(f"Expected {X_train.shape[1]}, got {len(feat)}")
                continue

            yhat = float(model.predict(np.asarray([feat], dtype=float))[0])
            predictions_dict[(target_state, year)] = yhat

    return predictions_dict, dta_dict

def create_output_dataframe(predictions_dict, dta_dict, treated_id=3):
    """
    Create final output DataFrame with predictions and gaps.

    Parameters:
    -----------
    predictions_dict : dict
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
        description='Run TabPFNv2 predictions with synbeats-style features'
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
        default='tabpfn_predictions_synbeats.csv',
        help='Output CSV filename (default: tabpfn_predictions_synbeats.csv)'
    )
    parser.add_argument(
        '--treated-id',
        type=int,
        default=3,
        help='ID of treated state to exclude (default: 3 for California)'
    )
    parser.add_argument(
        '--n-pcs',
        type=int,
        default=2,
        help='Number of PCA components (default: 2)'
    )

    args = parser.parse_args()

    # Setup device
    setup_device()

    # Load data
    print(f"Loading data from: {args.data_path}")
    dta = load_smoking_data(args.data_path)
    print(f"Loaded {len(dta)} observations")

    # Run predictions with synbeats features
    predictions_dict, dta_dict = run_tabpfn_predictions_synbeats(
        dta,
        treated_id=args.treated_id,
        n_pcs=args.n_pcs
    )

    # Create output DataFrame
    out = create_output_dataframe(predictions_dict, dta_dict, treated_id=args.treated_id)

    # Save to CSV
    output_path = args.output
    out.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    print(f"Output shape: {out.shape}")

if __name__ == "__main__":
    main()
