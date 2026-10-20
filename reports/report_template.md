# AirLens: Multi-Site Air Pollution Pattern Mining and Next-Hour Analytics

**Course:** Data Warehousing & Data Mining Lab (CS-416)  
**Student name / roll number:** [Fill in]  
**Instructor:** [Fill in]  
**Date:** [Fill in]

## 1. Abstract
Write 150–200 words describing the problem, dataset, methods, and actual results after running the project. Do not fill in imaginary values.

## 2. Problem Statement and Motivation
Explain why hourly pollution data can be studied by station, time, and weather conditions. State that the project analyzes historical Beijing data and is for education, not a live warning service.

## 3. Objectives
- Clean and profile multi-station air-quality data.
- Discover co-occurring pollutant/weather categories with Apriori.
- Segment pollution observations using K-Means.
- Estimate next-hour PM2.5 and classify a project-defined high-pollution target.
- Create a SQLite analytical warehouse and Streamlit interface.

## 4. Dataset Description
**Source:** UCI Machine Learning Repository, Beijing Multi-Site Air Quality dataset.  
Record the exact downloaded file, number of rows loaded, columns, stations, period, and citation. Add a column description table.

## 5. Tools and Environment
Record operating system, Python version, package versions, CPU/RAM, and IDE.

## 6. Methodology
### 6.1 Data acquisition and preprocessing
Describe duplicate station/timestamp removal, timestamp validation, invalid negative sensor values, and missing-value handling. Include before/after row counts and a missingness table.

### 6.2 Exploratory data analysis
Insert station PM2.5 plot, monthly trend, month/hour heatmap, and correlation heatmap. Describe only visible patterns supported by your figures.

### 6.3 Data warehouse
Describe `dim_station`, `dim_time`, `fact_air_quality`, and `station_month_summary`. Include the schema/ER diagram and one SQL query output.

### 6.4 Association rule mining
State sample size, minimum support, minimum confidence, and the number of itemsets/rules. Include the top 5–10 actual rules and explain support, confidence, and lift.

### 6.5 Clustering
State K, sample size, features, imputation and scaling choices. Include cluster profiles and explain each cluster using its measured statistics.

### 6.6 Classification
Describe the next-hour high-PM2.5 target, training-only 75th percentile threshold, chronological train/test split, features, model settings, confusion matrix, accuracy, precision, recall, and F1. Compare against the majority-class baseline.

### 6.7 Regression
Describe the one-hour-ahead PM2.5 target, features, chronological split, Random Forest settings, MAE, RMSE, R², and the persistence-baseline MAE/RMSE.

## 7. Results and Insights
Write 4–8 evidence-based findings from generated files. Each finding should point to a chart, table, rule, or model metric. Distinguish association from causation.

## 8. Dashboard
Insert screenshots of the Overview, Association Mining, Clusters, Forecast Evaluation, and Data Warehouse tabs.

## 9. Limitations
Mention the historic Beijing-only data, missing sensor readings, sampling used for computational efficiency, model uncertainty, and the fact that the project's “high” class is not an official AQI category.

## 10. Conclusion and Future Scope
Summarize what was found in your actual run. Possible future work: compare another public city dataset, test seasonal models, tune hyperparameters, and deploy with updated verified observations.

## 11. References
1. UCI Machine Learning Repository. *Beijing Multi-Site Air Quality Data*. Use the current citation details from the official dataset page.
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. mlxtend frequent patterns documentation: https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
4. Streamlit documentation: https://docs.streamlit.io/
