# Feature Engineering Comparison

This document compares the two feature engineering approaches available for TabPFNv2 predictions.

## Quick Comparison

| Aspect | Basic Window Features | Synbeats-Style Features |
|--------|----------------------|------------------------|
| **Script** | `run_tabpfn_predictions.py` | `run_tabpfn_predictions_synbeats.py` |
| **Complexity** | Simple | Advanced |
| **Features** | Raw cigarette sales in ±5 year windows | 15+ engineered features per state-year |
| **Panel Statistics** | No | Yes (means, std, PCA) |
| **Trend Modeling** | No | Yes (linear trends + residuals) |
| **Feature Dimension** | Variable (depends on #states × window) | Fixed set of features |
| **Speed** | Faster | Slower (more computation) |
| **Recommended For** | Quick baseline, simple approach | More sophisticated modeling |

## Basic Window Features

### What it uses:
- Raw cigarette sales from other control states
- Values from [year-5, year+5] window around each training year
- Past lags of the target state

### Example feature vector for state i at year 1980:
```
[
  state_1_1975, state_1_1976, ..., state_1_1985,  # State 1's sales from 1975-1985
  state_2_1975, state_2_1976, ..., state_2_1985,  # State 2's sales from 1975-1985
  ...
  state_i_1975, state_i_1976, state_i_1977, state_i_1978, state_i_1979  # Own past lags
]
```

### Advantages:
- ✅ Simple and interpretable
- ✅ Fast to compute
- ✅ Direct use of raw data
- ✅ Natural handling of temporal windows

### Disadvantages:
- ❌ High dimensional (many features)
- ❌ No explicit trend modeling
- ❌ Doesn't capture common factors across states

## Synbeats-Style Features

### What it uses:
For each state-year, computes:

**1. Time features:**
- `t`: Years since 1970
- `t2`: Squared time (captures acceleration)

**2. Within-state features (horizontal):**
- `lag1`, `lag2`, `lag3`: Past 1-3 years of sales
- `diff_lag`: Change in sales (lag1 - lag2)
- `mean_lag3`: Average of last 3 years

**3. Panel features (vertical):**
- `panel_mean`: Average sales across all control states this year
- `panel_sd`: Std deviation of sales across control states
- `pc1`, `pc2`: First 2 principal components from PCA

**4. Trend features:**
- `trend_pred`: State-specific linear trend prediction
- `residual`: Deviation from trend
- `lag1_resid`: Lagged residual

### Example feature vector for state i at year 1980:
```
[
  id,                    # State ID
  10, 100,              # t=10 (1980-1970), t²=100
  120.5, 118.2, 115.7,  # lag1, lag2, lag3
  2.3, 118.1,           # diff_lag, mean_lag3
  125.4, 15.2,          # panel_mean, panel_sd
  -0.32, 1.45,          # pc1, pc2
  112.8, 7.7, 6.3       # trend_pred, residual, lag1_resid
]
```

### Advantages:
- ✅ Rich feature set captures multiple aspects
- ✅ Lower dimensional than basic features
- ✅ Explicit trend and residual modeling
- ✅ Captures common factors via PCA
- ✅ Both horizontal (time) and vertical (panel) information

### Disadvantages:
- ❌ More complex to understand
- ❌ Slower to compute
- ❌ Requires feature recomputation for recursive forecasting
- ❌ More hyperparameters (e.g., n_pcs)

## When to Use Which?

### Use Basic Window Features if:
- You want a quick baseline
- You prefer simplicity and interpretability
- You want to minimize computation time
- You're doing initial exploratory analysis

### Use Synbeats-Style Features if:
- You want potentially better prediction accuracy
- You value capturing trends and common factors
- You're willing to trade speed for sophistication
- You want to incorporate domain knowledge about panel data structure

## Implementation Details

### Placebo Test Considerations

Both approaches properly handle placebo tests by:
1. **Excluding California** (treated state) from all computations
2. **Excluding the current target state** from panel statistics when predicting for that state

For synbeats features, this means:
- When predicting for state i, both California and state i are excluded from:
  - Panel means and standard deviations
  - PCA computation
- This ensures no "leakage" of information from the target state

### Recursive Forecasting

**Basic approach:** Directly updates the prediction dictionary and uses those predictions as lags for future years.

**Synbeats approach:** Uses features from the pre-computed feature matrix, which means features are based on actual observations rather than recursively updated predictions for control states.

## Recommendations

1. **Start with basic features** to establish a baseline
2. **Run synbeats features** to see if additional sophistication helps
3. **Compare results** using the gap and RMSE metrics
4. **Consider ensemble** approaches combining both methods

## Output Files

Both scripts produce identical output format:
- `id`: State identifier
- `year`: Year
- `predicted`: Model prediction
- `obs`: Observed value
- `gap`: obs - predicted

This makes it easy to compare results between approaches.
