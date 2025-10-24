Task 2 – Advanced Analytics: Sales Forecasting & KPI Anomaly Detection
Project Status: Completed

Objective
This project aims to develop a robust data pipeline to forecast future sales and automatically detect historical anomalies in key performance indicators (KPIs). The models analyze sales data aggregated by product, region, or overall monthly performance. The final outputs include forecast data, anomaly reports, visualizations, and actionable insights designed to integrate into a business intelligence dashboard.

Data Sources & Preparation
Source: The primary data source is a CSV file (sales_data.csv) containing transactional sales records. The required columns are: date, region, product, units_sold, and unit_price.
Preparation Pipeline:
Loading: The raw data is loaded into a pandas DataFrame.
Data Typing: The date column is converted to a datetime object.
Aggregation: Data is grouped by the chosen level (product, region, or overall) and aggregated into monthly totals for units_sold. This creates distinct time series for analysis.
Methodology
Forecasting
To ensure accuracy, a comparative approach was used. Two models were evaluated for each time series, and the best-performing one was selected automatically.

Seasonal Baseline (Naive Model): This model forecasts future values based on the average of the last 12 months of historical data. It serves as a simple benchmark for performance.
Advanced Model (e.g., SARIMA): A statistical model that excels at capturing complex trend and seasonality patterns. The script uses SARIMA from the statsmodels library, but can be adapted to use other powerful models like Auto-ARIMA from the pmdarima library.
Quality Check & Model Selection: The models were compared using Mean Absolute Error (MAE) on a validation set. The model with the lower MAE for each series was chosen to generate the final forecast, ensuring the most accurate prediction possible.

forecast_A
Anomaly Detection
Anomalies (unusual spikes or drops) in historical sales data were identified using the Isolation Forest algorithm. This unsupervised learning model is highly effective because it does not assume the data follows a specific distribution and is efficient at identifying outliers.

forecast_Overall
How to Reproduce the Analysis
Follow these steps to set up the environment and run the project.

1. Prerequisites
Python 3.8+
Git
2. Clone the Repository
git clone <your-github-repo-link>
cd <repository-folder>
3. Set up a Virtual Environment
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Installation
Create a requirements.txt file with the following content.

# requirements.txt
pandas
scikit-learn
statsmodels
matplotlib
pmdarima
Now, run the installation command:

pip install -r requirements.txt
Important Note on Installation Errors: Some libraries, like pmdarima, may fail to install if you don't have the necessary C/C++ compilers on your system. If you see a "Failed to build wheel" error, you must install the development tools for your operating system:

Windows: Install Microsoft C++ Build Tools from the Visual Studio website (select the "Desktop development with C++" workload).
macOS: Install Xcode Command Line Tools by running xcode-select --install in your terminal.
Linux (Debian/Ubuntu): Install build-essential by running sudo apt-get install build-essential python3-dev.
5. Run the Analysis Script
Place your sales data (e.g., sales_data.csv) in the root folder and run the main script. The script will guide you through the required inputs.

python run_forecasting.py
The script will generate all outputs in the specified reports and images folders.
