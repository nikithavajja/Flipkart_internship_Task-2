# Original Code by Vajja Sri Nikitha 
# Built for Flipkart Task-2 Sales Forecasting & KPI Anomaly Detection 
# Task Given :-  Build models to forecast sales for the next 3–6 months and 
#                detect anomalies in key KPIs across product and region segments, 
#                generating clear charts, CSV outputs, and a concise insights report 
#                that can plug into the reporting/dashboard workflow.
# Date(Last Updated): 24-10-2025

"""
Sales Forecasting & Anomaly Detection Tool
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Anomaly detection will be limited.")

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. SARIMA forecasting will be unavailable.")


class ColoredOutput:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

    @staticmethod
    def print_header(text):
        print(f"\n{ColoredOutput.HEADER}{ColoredOutput.BOLD}{text}{ColoredOutput.END}")
    
    @staticmethod
    def print_success(text):
        print(f"{ColoredOutput.GREEN}[SUCCESS] {text}{ColoredOutput.END}")
    
    @staticmethod
    def print_error(text):
        print(f"{ColoredOutput.RED}[ERROR] {text}{ColoredOutput.END}")
    
    @staticmethod
    def print_info(text):
        print(f"{ColoredOutput.BLUE}[INFO] {text}{ColoredOutput.END}")
    
    @staticmethod
    def print_warning(text):
        print(f"{ColoredOutput.YELLOW}[WARNING] {text}{ColoredOutput.END}")


def ensure_dirs(paths):
    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            ColoredOutput.print_success(f"Directory ready: {p}")
        except Exception as e:
            ColoredOutput.print_error(f"Could not create directory {p}: {e}")
            sys.exit(1)


def get_file_path():
    while True:
        print("\n" + "="*60)
        file_path = input("Enter the path to your sales CSV file: ").strip()
        
        if not file_path:
            ColoredOutput.print_error("Path cannot be empty.")
            continue
            
        if not os.path.exists(file_path):
            ColoredOutput.print_error(f"File not found: {file_path}")
            retry = input("Try again? (yes/no): ").lower()
            if retry not in ['yes', 'y', '']:
                sys.exit(0)
            continue
            
        if not file_path.lower().endswith('.csv'):
            ColoredOutput.print_warning("File doesn't end with .csv")
            proceed = input("Continue anyway? (yes/no): ").lower()
            if proceed not in ['yes', 'y']:
                continue
        
        ColoredOutput.print_success(f"File found: {file_path}")
        return file_path


def select_column(columns, prompt, allow_skip=False):
    print("\n" + "-"*60)
    print(f"{prompt}")
    print("-"*60)
    
    for i, col in enumerate(columns, 1):
        print(f"  {i:2d}. {col}")
    
    if allow_skip:
        print("   0. Skip")
    
    while True:
        try:
            choice = input(f"\nEnter number (1-{len(columns)}): ").strip()
            
            if not choice:
                ColoredOutput.print_error("Input cannot be empty")
                continue
            
            choice_num = int(choice)
            
            if allow_skip and choice_num == 0:
                return None
            
            if 1 <= choice_num <= len(columns):
                selected = columns[choice_num - 1]
                ColoredOutput.print_success(f"Selected: {selected}")
                return selected
            else:
                ColoredOutput.print_error(f"Please enter a number between 1 and {len(columns)}")
        except ValueError:
            ColoredOutput.print_error("Please enter a valid number")


def load_and_clean_data(path, date_col, sales_col):
    ColoredOutput.print_header("Loading and Cleaning Data")
    
    try:
        df = pd.read_csv(path)
        initial_rows = len(df)
        ColoredOutput.print_info(f"Loaded {initial_rows:,} rows")
        
        try:
            # --- THIS IS THE FIX ---
            # Added dayfirst=True to correctly parse DD/MM/YYYY formats
            df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
            ColoredOutput.print_success(f"Parsed date column: {date_col}")
            
        except Exception as e:
            ColoredOutput.print_error(f"Could not parse dates in column '{date_col}'")
            ColoredOutput.print_error(f"This column may contain mixed or invalid date formats.")
            ColoredOutput.print_error("Please restart and select a column with actual dates (e.g., '2024-01-15', '01/15/2024')")
            sys.exit(1)
        
        df[sales_col] = pd.to_numeric(df[sales_col], errors='coerce')
        df = df.dropna(subset=[date_col, sales_col])
        
        negative_count = (df[sales_col] < 0).sum()
        if negative_count > 0:
            ColoredOutput.print_warning(f"Removing {negative_count} rows with negative sales")
            df = df[df[sales_col] >= 0]
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        if removed > 0:
            ColoredOutput.print_info(f"Removed {removed:,} invalid rows ({removed/initial_rows*100:.1f}%)")
        
        ColoredOutput.print_success(f"Final dataset: {final_rows:,} rows")
        
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        ColoredOutput.print_info(f"Date range: {min_date.date()} to {max_date.date()}")
        
        return df
        
    except FileNotFoundError:
        ColoredOutput.print_error(f"File not found: {path}")
        sys.exit(1)
    except Exception as e:
        ColoredOutput.print_error(f"Error loading data: {e}")
        sys.exit(1)


def aggregate_data(df, level, date_col, sales_col, series_col):
    ColoredOutput.print_header(f"Aggregating Data: {level.upper()} level")
    
    try:
        if level == "overall":
            agg = df.groupby(pd.Grouper(key=date_col, freq='MS'))[sales_col].sum().reset_index()
            agg['series'] = 'Overall'
        else:
            agg = df.groupby([
                pd.Grouper(key=date_col, freq='MS'),
                series_col
            ])[sales_col].sum().reset_index()
            agg = agg.rename(columns={series_col: 'series'})
        
        agg = agg.rename(columns={date_col: 'date', sales_col: 'units_sold'})
        agg = agg[agg['units_sold'] > 0]
        
        num_series = agg['series'].nunique()
        ColoredOutput.print_success(f"Created {num_series} time series")
        
        for series_name in agg['series'].unique():
            series_data = agg[agg['series'] == series_name]
            ColoredOutput.print_info(f"  {series_name}: {len(series_data)} months")
        
        return agg
        
    except Exception as e:
        ColoredOutput.print_error(f"Aggregation failed: {e}")
        sys.exit(1)


def evaluate_and_forecast(series_df, horizon, series_name):
    series_df = series_df.sort_values('date').copy()
    series_df = series_df.set_index('date')
    y = series_df['units_sold'].astype(float)
    
    if len(y) < 12:
        ColoredOutput.print_warning(f"  {series_name}: Only {len(y)} months. Using simple average.")
        forecast = [np.mean(y)] * horizon
        return forecast, "Simple Average", None
    
    if len(y) >= 24 and SKLEARN_AVAILABLE:
        train = y[:-horizon]
        test = y[-horizon:]
        
        baseline_forecast = [np.mean(train.tail(12))] * horizon
        baseline_mae = mean_absolute_error(test, baseline_forecast)
        
        sarima_mae = float('inf')
        sarima_forecast = None
        
        if STATSMODELS_AVAILABLE:
            try:
                model = SARIMAX(
                    train,
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                result = model.fit(disp=False, maxiter=100)
                sarima_forecast = result.get_forecast(steps=horizon).predicted_mean.values
                sarima_mae = mean_absolute_error(test, sarima_forecast)
            except Exception as e:
                ColoredOutput.print_warning(f"  SARIMA failed for {series_name}: {str(e)[:50]}")
        
        if sarima_mae < baseline_mae and sarima_forecast is not None:
            try:
                model = SARIMAX(
                    y,
                    order=(1, 1, 1),
                    seasonal_order=(0, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                result = model.fit(disp=False, maxiter=100)
                forecast = result.get_forecast(steps=horizon).predicted_mean.values.tolist()
                ColoredOutput.print_info(f"  {series_name}: SARIMA (MAE: {sarima_mae:.2f})")
                return forecast, "SARIMA", round(sarima_mae, 2)
            except:
                pass
        
        forecast = [np.mean(y.tail(12))] * horizon
        ColoredOutput.print_info(f"  {series_name}: Seasonal Baseline (MAE: {baseline_mae:.2f})")
        return forecast, "Seasonal Baseline", round(baseline_mae, 2)
    
    else:
        forecast = [np.mean(y.tail(min(12, len(y))))] * horizon
        ColoredOutput.print_info(f"  {series_name}: Average (insufficient data for validation)")
        return forecast, "Average", None


def detect_anomalies(series_df, series_name):
    series_df = series_df.sort_values('date').copy()
    
    if not SKLEARN_AVAILABLE or len(series_df) < 12:
        series_df['anomaly'] = 0
        return series_df
    
    try:
        X = series_df['units_sold'].values.reshape(-1, 1)
        iso = IsolationForest(contamination=0.05, random_state=42)
        labels = iso.fit_predict(X)
        series_df['anomaly'] = (labels == -1).astype(int)
        
        num_anomalies = series_df['anomaly'].sum()
        if num_anomalies > 0:
            ColoredOutput.print_warning(f"  Found {num_anomalies} anomalies in {series_name}")
    except Exception as e:
        ColoredOutput.print_warning(f"  Anomaly detection failed for {series_name}: {e}")
        series_df['anomaly'] = 0
    
    return series_df


def plot_forecast(hist_df, forecast_df, series_name, output_dir):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(hist_df['date'], hist_df['units_sold'], 
                label='Historical', color='#2E86AB', linewidth=2, marker='o', markersize=4)
        plt.plot(forecast_df['date'], forecast_df['forecast_units'],
                label='Forecast', color='#A23B72', linewidth=2, linestyle='--', marker='s', markersize=5)
        
        plt.title(f'Sales Forecast: {series_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Units Sold', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        filename = f"forecast_{series_name.replace(' ', '_').replace('/', '_')}.png"
        filepath = Path(output_dir) / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        ColoredOutput.print_warning(f"Could not create plot for {series_name}: {e}")


def generate_summary(model_performance, anomalies_df, level, horizon, export_dir):
    lines = ["# Sales Forecasting Analysis Report\n"]
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"**Aggregation Level:** {level.title()}\n")
    lines.append(f"**Forecast Horizon:** {horizon} months\n")
    lines.append("---\n")
    
    lines.append("## Model Performance\n")
    lines.append("The best-performing model was selected for each series based on validation MAE.\n")
    
    if model_performance:
        perf_df = pd.DataFrame(model_performance)
        lines.append("\n" + perf_df.to_markdown(index=False) + "\n")
    else:
        lines.append("No model performance data available.\n")
    
    lines.append("\n## Detected Anomalies\n")
    recent_anomalies = anomalies_df[anomalies_df['anomaly'] == 1].sort_values('date', ascending=False).head(15)
    
    if not recent_anomalies.empty:
        lines.append("The following unusual data points were detected:\n")
        for _, row in recent_anomalies.iterrows():
            lines.append(f"- **{row['series']}** on {row['date'].date()}: {int(row['units_sold'])} units\n")
    else:
        lines.append("No significant anomalies detected.\n")
    
    summary_path = Path(export_dir) / 'analysis_summary.md'
    summary_path.write_text(''.join(lines))
    return summary_path


def main():
    ColoredOutput.print_header("=" * 60)
    ColoredOutput.print_header("SALES FORECASTING & ANOMALY DETECTION TOOL")
    ColoredOutput.print_header("=" * 60)
    
    file_path = get_file_path()
    
    try:
        headers = pd.read_csv(file_path, nrows=0).columns.tolist()
        ColoredOutput.print_success(f"Found {len(headers)} columns in dataset")
    except Exception as e:
        ColoredOutput.print_error(f"Could not read CSV headers: {e}")
        sys.exit(1)
    
    date_col = select_column(headers, "Select your DATE column")
    sales_col = select_column(headers, "Select your SALES/UNITS column")
    
    print("\n" + "-"*60)
    print("Choose aggregation level:")
    print("  1. Overall (total sales)")
    print("  2. By Product")
    print("  3. By Region")
    print("-"*60)
    
    while True:
        level_choice = input("Enter choice (1-3): ").strip()
        if level_choice == '1':
            level = 'overall'
            series_col = None
            break
        elif level_choice == '2':
            level = 'product'
            series_col = select_column(headers, "Select PRODUCT column")
            break
        elif level_choice == '3':
            level = 'region'
            series_col = select_column(headers, "Select REGION column")
            break
        else:
            ColoredOutput.print_error("Please enter 1, 2, or 3")
    
    while True:
        try:
            horizon_input = input("\nForecast horizon in months (default 6): ").strip()
            horizon = int(horizon_input) if horizon_input else 6
            if horizon < 1 or horizon > 24:
                ColoredOutput.print_error("Please enter a value between 1 and 24")
                continue
            ColoredOutput.print_success(f"Forecast horizon: {horizon} months")
            break
        except ValueError:
            ColoredOutput.print_error("Please enter a valid number")
    
    export_dir = input("\nReports directory (default 'reports'): ").strip() or 'reports'
    image_dir = input("Charts directory (default 'charts'): ").strip() or 'charts'
    
    ensure_dirs([export_dir, image_dir])
    
    df = load_and_clean_data(file_path, date_col, sales_col)
    aggregated = aggregate_data(df, level, date_col, sales_col, series_col)
    
    ColoredOutput.print_header("Generating Forecasts")
    
    forecast_results = []
    anomaly_results = []
    model_performance = []
    
    for series_name, series_data in aggregated.groupby('series'):
        preds, model_name, mae = evaluate_and_forecast(series_data, horizon, series_name)
        
        if mae is not None:
            model_performance.append({
                'Series': series_name,
                'Model': model_name,
                'Validation MAE': mae
            })
        
        last_date = series_data['date'].max()
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=horizon, freq='MS')
        
        forecast_df = pd.DataFrame({
            'series': series_name,
            'date': future_dates,
            'forecast_units': preds
        })
        forecast_results.append(forecast_df)
        
        plot_forecast(series_data, forecast_df, series_name, image_dir)
        
        series_with_anomalies = detect_anomalies(series_data, series_name)
        anomaly_results.append(series_with_anomalies)
    
    ColoredOutput.print_header("Saving Results")
    
    forecast_combined = pd.concat(forecast_results, ignore_index=True)
    anomalies_combined = pd.concat(anomaly_results, ignore_index=True)
    
    forecast_path = Path(export_dir) / 'forecasts.csv'
    anomalies_path = Path(export_dir) / 'anomalies.csv'
    
    forecast_combined.to_csv(forecast_path, index=False)
    anomalies_combined.to_csv(anomalies_path, index=False)
    
    summary_path = generate_summary(model_performance, anomalies_combined, level, horizon, export_dir)
    
    ColoredOutput.print_header("ANALYSIS COMPLETE")
    print(f"\nForecasts:  {forecast_path}")
    print(f"Anomalies:  {anomalies_path}")
    print(f"Summary:    {summary_path}")
    print(f"Charts:     {image_dir}/\n")
    
    ColoredOutput.print_success("All outputs saved successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        ColoredOutput.print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
