# AirLens — Air-Quality Pattern Mining and Next-Hour Analytics

**Course alignment:** Data Warehousing & Data Mining Lab (CS-416)  
**Language:** Python  
**Main tools:** pandas, scikit-learn, mlxtend/Apriori, SQLite, Streamlit

AirLens examines historical air-quality observations from the UCI Beijing Multi-Site Air Quality dataset. It demonstrates data cleaning, exploratory analysis, association-rule mining, K-Means clustering, time-ordered classification/regression, a small star-schema-style SQLite warehouse, and an interactive dashboard.

> **Scope and safety note:** This is an educational analysis of historical Beijing data. It is not a live monitoring service, an official AQI calculator, or a public-health warning tool. Model outputs are estimates and should not be used for health or emergency decisions.

## 1. Download the dataset

1. Open the [official UCI Beijing Multi-Site Air Quality page](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data).
2. Download the ZIP listed as `PRSA2017_Data_20130301-20170228.zip`.
3. Copy the ZIP unchanged into this folder: `AirLens/data/raw/`.
4. You do **not** need to unzip it; the loader reads the station CSV files inside the archive. Extracted station CSVs in `data/raw/` are supported too.

The raw dataset is not bundled with this code package. Keep the original data file unchanged for reproducibility and follow the license and citation on the UCI page.

## 2. Requirements

- Windows 10/11
- Python 3.11–3.13 (64-bit recommended)
- VS Code (recommended)
- Internet connection to download the dataset and Python packages
- Approximately 2–4 GB of free disk space is a comfortable working allowance; the exact output size depends on the data and generated SQLite warehouse.

Check Python:

```powershell
python --version
```

If Python is not found, install a supported Python version from [python.org](https://www.python.org/downloads/) and select **Add Python to PATH** during installation.

## 3. Set up in VS Code (Windows PowerShell)

Open the `AirLens` folder in VS Code. Select **Terminal → New Terminal**, then run each command:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, select **Command Prompt** as the terminal profile and run:

```bat
.venv\Scripts\activate.bat
```

Verify packages:

```powershell
python -c "import pandas, sklearn, mlxtend, streamlit; print('AirLens setup OK')"
```

## 4. Run the pipeline

Confirm the ZIP is in `data/raw/`, and the virtual environment is active. Then run:

```powershell
python run_pipeline.py
```

The program performs the following stages:

1. Reads the PRSA station CSV files.
2. Validates timestamps, removes duplicate station/timestamp rows, and changes invalid negative pollutant concentrations/wind speeds to missing values. Missing readings are retained rather than silently fabricated.
3. Saves the cleaned CSV and missingness report.
4. Produces exploratory charts and summary statistics.
5. Creates a SQLite warehouse with `dim_station`, `dim_time`, `fact_air_quality`, and `station_month_summary` tables.
6. Runs Apriori association mining on sampled categorical station-hour observations.
7. Fits K-Means on a reproducible sample and assigns pollution-regime cluster IDs.
8. Trains a one-hour-ahead PM2.5 regressor and a high-PM2.5 classifier using a chronological split. Numeric/categorical imputation and encoding are fitted inside the training pipeline, and the classification threshold is calculated from training targets only.

The run may take several minutes on a laptop. The model training set is capped at 90,000 rows for practical CPU/memory usage. K-Means and Apriori also use reproducible samples; these sample sizes are set in `airlens/analysis.py` and can be adjusted.

## 5. Start the dashboard

After the pipeline has completed:

```powershell
python -m streamlit run app.py
```

Streamlit opens a local page in your browser. Use the sidebar to filter by station/date and pick a pollutant. Tabs show trends, association rules, cluster profiles, forecasting evaluation, and example warehouse queries. Stop the server with `Ctrl+C` in the terminal.

If the browser page opens before processing is complete, the app displays a message directing you to run the pipeline first.

## 6. Project structure

```text
AirLens/
├── airlens/
│   ├── config.py             # Column lists and directory configuration
│   ├── data_io.py            # Read the official ZIP or extracted CSVs
│   ├── preprocessing.py      # Cleaning, timestamp creation, reports
│   ├── analysis.py           # EDA, Apriori, K-Means
│   ├── warehouse.py          # SQLite star-schema-style warehouse
│   ├── models.py             # Time-ordered forecast models and metrics
│   └── pipeline.py           # Runs all stages in sequence
├── data/
│   ├── raw/                  # Put the official UCI ZIP here
│   └── processed/            # Generated cleaned observations
├── outputs/                  # Charts, metrics, models, DB and reports
├── reports/
│   ├── report_template.md    # Lab report outline
│   └── viva_questions.md     # Practice viva questions
├── app.py                    # Streamlit dashboard
├── run_pipeline.py           # Main entry point
├── requirements.txt
└── README.md
```

## 7. Important generated outputs

| Output | What it is for |
|---|---|
| `data/processed/cleaned_observations.csv` | Cleaned analytical dataset, with missing measurements preserved |
| `outputs/preprocessing_summary.json` | Cleaning counts, date range, station names, missingness |
| `outputs/missingness_report.csv` | Missing values by column |
| `outputs/monthly_pm25.png` | Monthly average PM2.5 pattern |
| `outputs/month_hour_heatmap.png` | Mean PM2.5 by month/hour |
| `outputs/correlation_heatmap.png` | Pairwise numeric correlations |
| `outputs/frequent_itemsets.csv` | Frequent combinations found by Apriori |
| `outputs/association_rules.csv` | Support, confidence and lift for association rules |
| `outputs/cluster_profiles.csv` | Per-cluster mean/median/count summaries |
| `outputs/forecast_predictions.csv` | Test-period actual and predicted values/classes |
| `outputs/model_metrics.json` | Chronological split details and evaluation metrics |
| `outputs/airlens_warehouse.db` | SQLite analytical warehouse |
| `outputs/*joblib` | Trained reusable preprocessing/model pipelines |

## 8. How to interpret methods

### Preprocessing
Duplicate keys are defined as the same station and timestamp. Negative pollutant readings and negative wind speeds are marked as missing, not treated as real measurements. We do not globally interpolate the data before forecasting: imputation for the models is fitted on training rows only.

### Association rules
Rows represent station-hour observations, and items are categories such as `PM2.5=High`, `Wind=Low`, season and station. Support is the fraction of sampled transactions containing the full itemset. Confidence is the conditional fraction containing a consequent when an antecedent appears. Lift above 1 indicates a positive association relative to independence, but it does not establish causation.

### Clustering
K-Means is fitted after median imputation and standardization. The default K is 4. The labels (0, 1, 2, 3) are arbitrary; use cluster profiles and median PM2.5 to describe patterns. Cluster IDs are not official air-quality categories.

### One-hour-ahead forecasting
The model target is PM2.5 at the next clock hour for the same station. Rows are only used when the following station observation is exactly one hour later. The data split is chronological by target timestamp; the later 20% is the test period. Features are from the current observation and past PM2.5 lags. The baseline forecast carries the current PM2.5 value forward. Regression MAE/RMSE should be compared with that persistence baseline.

The classifier label “high” means target PM2.5 exceeds the 75th percentile of the training target distribution. It is a project-specific label, not a regulatory AQI band.

### Warehouse
- `dim_station`: monitoring station dimension.
- `dim_time`: hour/time dimension with year, month, day, hour, weekday and season.
- `fact_air_quality`: numeric measurements keyed to station and time.
- `station_month_summary`: pre-aggregated monthly measures for OLAP-style queries.

## 9. Troubleshooting

**`No matching PRSA CSV data was found`**  
Check that the official ZIP is directly inside `data/raw/`, and that the filename ends in `.zip`. Do not put a screenshot or a webpage shortcut in this folder.

**`python` is not recognized**  
Try `py --version`. If it works, use `py -m venv .venv`, then activate the environment as documented. Otherwise install Python and enable PATH.

**PowerShell script execution is blocked**  
Use the VS Code Command Prompt terminal and run `.venv\Scripts\activate.bat` instead.

**`ModuleNotFoundError`**  
Activate `.venv`, then run `pip install -r requirements.txt` again. Make sure VS Code is using the `.venv` interpreter (Ctrl+Shift+P → Python: Select Interpreter).

**The pipeline appears slow**  
The warehouse and model stages process hundreds of thousands of rows. Wait for the current stage; do not start multiple copies. For a faster demonstration, lower `sample_limit` in `airlens/analysis.py` or `max_train_rows` in `airlens/models.py`.

**Apriori exports no rules**  
This can happen after support/confidence filtering. Inspect `frequent_itemsets.csv`; for experimentation, lower `min_support` from `0.03` to `0.02` or lower `min_threshold` from `0.45` to `0.35`. Document parameter changes in your report.

**Dashboard says outputs are missing**  
Run `python run_pipeline.py` and wait for it to finish before launching Streamlit.

**Unexpected model scores**  
Do not invent or hard-code results. Record actual metrics from `outputs/model_metrics.json`, compare against the persistence/majority baselines, and explain the limitations of an older regional dataset.

## 10. Suggested project report and citation

Use `reports/report_template.md` as the basis for your final lab report. Cite the UCI Machine Learning Repository dataset page and include your own run date, actual metrics, and screenshots. Do not state a result before the code has produced it.

**Dataset reference:** Chen, S. (2017). *Beijing Multi-Site Air Quality Data*. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G (verify current citation details on the dataset page before final submission).
