import pandas as pd
import numpy as np
import os

# --- Configuration ---
RESULTS_DIR_MINMAX = 'results_tuning_v3' # Directory with MinMaxScaler results
RESULTS_DIR_LOGSTD = 'results-v2'      # Directory with Log+StandardScaler results

SUMMARY_FILE_MINMAX = os.path.join(RESULTS_DIR_MINMAX, 'tuning_summary.csv')
SUMMARY_FILE_LOGSTD = os.path.join(RESULTS_DIR_LOGSTD, 'tuning_summary.csv')

N_TOP_MODELS = 5 # Number of top models to compare from each run

# --- 1. Load Summary Data ---
try:
    df_minmax = pd.read_csv(SUMMARY_FILE_MINMAX)
    print(f"Loaded MinMaxScaler results from: {SUMMARY_FILE_MINMAX} ({len(df_minmax)} rows)")
except FileNotFoundError:
    print(f"Error: File not found at {SUMMARY_FILE_MINMAX}")
    df_minmax = None

try:
    df_logstd = pd.read_csv(SUMMARY_FILE_LOGSTD)
    print(f"Loaded Log+StandardScaler results from: {SUMMARY_FILE_LOGSTD} ({len(df_logstd)} rows)")
except FileNotFoundError:
    print(f"Error: File not found at {SUMMARY_FILE_LOGSTD}")
    df_logstd = None

if df_minmax is None or df_logstd is None:
    print("Cannot proceed without both summary files.")
    exit()

# --- 2. Basic Cleaning ---
df_minmax = df_minmax[df_minmax['status'] == 'Success'].copy()
df_logstd = df_logstd[df_logstd['status'] == 'Success'].copy()
print(f"MinMaxScaler successful runs: {len(df_minmax)}")
print(f"Log+Std successful runs: {len(df_logstd)}")

# Add scaling method identifier
df_minmax['scaling'] = 'MinMax'
df_logstd['scaling'] = 'LogStd'

# Define metrics to analyze
metrics = ['test_rmse', 'test_mae', 'test_r2']

# --- 3. Overall Metric Comparison ---
print("\n--- Overall Metric Comparison (Test Set - Original Scale) ---")

stats_minmax = df_minmax[metrics].agg(['min', 'mean', 'median', 'max'])
stats_logstd = df_logstd[metrics].agg(['min', 'mean', 'median', 'max'])

print("\nMinMaxScaler Stats:")
print(stats_minmax)
print("\nLog+StandardScaler Stats:")
print(stats_logstd)

# --- 4. Identify Best Models ---
# Sort primarily by RMSE (ascending), then MAE (ascending), then R2 (descending)
sort_columns = ['test_rmse', 'test_mae', 'test_r2']
ascending_order = [True, True, False]

top_minmax = df_minmax.sort_values(by=sort_columns, ascending=ascending_order).head(N_TOP_MODELS)
top_logstd = df_logstd.sort_values(by=sort_columns, ascending=ascending_order).head(N_TOP_MODELS)

# --- 5. Detailed Comparison of Best Models ---
print(f"\n--- Top {N_TOP_MODELS} Models Comparison ---")

# Select relevant columns for comparison display
display_cols = [
    'combination_id', 'scaling',
    'test_rmse', 'test_mae', 'test_r2',
    'num_lstm_layers', 'lstm_units', 'num_dense_layers', 'dense_units',
    'training_time_s', 'best_epoch'
]

# Combine the top models for easier viewing
df_top_comparison = pd.concat([top_minmax[display_cols], top_logstd[display_cols]], ignore_index=True)

print(df_top_comparison.to_string()) # Use to_string to print full table

print("\n--- Analysis Guidance ---")
print("1. Review the 'Overall Metric Comparison' above.")
print("   - Did LogStd achieve lower min/mean/median RMSE and MAE?")
print("   - Did LogStd achieve higher max/mean/median R2?")

print("\n2. Review the 'Top Models Comparison' table.")
print(f"   - Compare the best metric values (e.g., lowest test_rmse) achieved by each scaling method.")
print(f"   - Note the complexity (layers/units) of the best models from each method.")

print("\n3. **VISUALLY INSPECT PLOTS (Your Task):**")
print(f"   - Go to the results directories: '{RESULTS_DIR_MINMAX}' and '{RESULTS_DIR_LOGSTD}'.")
print(f"   - For the top {N_TOP_MODELS} models listed in the table above (using 'combination_id'):")
print(f"     a) Open the corresponding plots in 'prediction_scatter_loglog/'.")
print(f"     b) **CRITICAL:** Compare the log-log plots side-by-side. Is the scatter around the y=x line visibly reduced and less biased for the 'LogStd' models, especially for AMR values < 0.05?")
print(f"     c) Briefly check the 'prediction_scatter_linear/' plots for the same models.")
print(f"     d) Check the 'loss/' plots. Do they show reasonable convergence?")

print("\n4. Synthesize:")
print(f"   - Based on BOTH the metrics AND your visual inspection, which scaling method ('MinMax' or 'LogStd') produced better, more reliable models for this specific problem?")
print(f"   - Identify the single best overall 'combination_id' considering performance, visual fit, and complexity.")