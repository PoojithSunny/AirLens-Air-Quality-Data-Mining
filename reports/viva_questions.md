# AirLens Viva Questions and Short Answer Points

1. **What is data mining?** Extracting useful patterns, relationships, groupings, or predictive signals from data.
2. **Why did you choose this project?** It combines environmental time-series analysis with several syllabus methods and uses a documented public dataset.
3. **What is preprocessing?** Validating, cleaning, transforming and preparing raw observations for reliable analysis.
4. **Why not replace all missing sensor values with zero?** Zero can be a valid measurement and would falsely imply an observed zero when the value is actually missing.
5. **Why remove duplicate station/timestamp rows?** They would double-count one station-hour observation.
6. **Why convert the date components into a timestamp?** It supports chronological sorting, temporal aggregation, lag features, and time-based splitting.
7. **What is EDA?** Exploratory data analysis uses summary statistics and plots to understand distributions, quality, trends and relationships.
8. **What is association rule mining?** It finds items that frequently co-occur in transactions.
9. **Define support.** Fraction of transactions containing an itemset.
10. **Define confidence.** Fraction of transactions containing an antecedent that also contain the consequent.
11. **Define lift.** Confidence divided by the consequent's support; lift above 1 indicates a positive association versus independence.
12. **Does a high lift prove that low wind causes pollution?** No. Association is not causation; confounding variables and the observational design matter.
13. **Why discretize measurements for Apriori?** Classic Apriori expects item-presence transactions; continuous readings must first be converted into categories.
14. **What is clustering?** An unsupervised method that groups records by similarity without using a target label.
15. **Why use K-Means?** It is a common, interpretable baseline for grouping scaled numerical observations.
16. **Why standardize before K-Means?** Distance-based clustering is sensitive to differences in measurement scales.
17. **What does K represent?** The chosen number of clusters; this project uses K=4 as a starting point, not as a proven optimum.
18. **Are cluster IDs ordinal?** No. IDs are arbitrary labels; profiles are needed to interpret them.
19. **What is classification?** Predicting a discrete class label, here a project-defined next-hour high-PM2.5 category.
20. **What is regression?** Predicting a continuous value, here next-hour PM2.5 concentration.
21. **What is Random Forest?** An ensemble of decision trees whose predictions are aggregated to improve robustness.
22. **What is a confusion matrix?** A table comparing actual and predicted classes, showing correct and incorrect predictions by class.
23. **Precision vs recall?** Precision measures the proportion of predicted positives that are positive; recall measures the proportion of actual positives detected.
24. **What is F1-score?** The harmonic mean of precision and recall.
25. **What does MAE mean?** The average absolute prediction error in target units.
26. **What does RMSE mean?** The square root of mean squared error, more sensitive to large errors than MAE.
27. **What is R²?** A score comparing model error with the mean-target baseline; it can be negative on test data.
28. **Why use a chronological split?** A random split can place later observations in training and earlier observations in testing, making a time forecast look unrealistically easy.
29. **What is data leakage?** When training uses information that would not legitimately be available at prediction time, inflating evaluation results.
30. **How does this project reduce leakage?** It targets the next clock hour, splits by target timestamp, creates lag features from past observations, and fits imputation/encoding on training rows only.
31. **What is the persistence baseline?** Predicting that the next-hour PM2.5 equals the current-hour PM2.5.
32. **Why compare against a baseline?** A complex model should show value beyond a simple reference method.
33. **What is a data warehouse?** A structured analytical store integrating data for querying, reporting, and analysis.
34. **What is a fact table?** It stores measurable observations or events, linked to dimensions.
35. **What are dimension tables?** They describe the context of facts, such as station and time.
36. **What is ETL?** Extract, Transform, Load—the process of reading source data, cleaning/transforming it, and loading a target store.
37. **What are OLAP operations?** Analytical operations such as roll-up, drill-down, slice, dice, and pivot.
38. **What does Streamlit do?** It turns Python scripts and model outputs into an interactive web interface.
39. **What is the biggest limitation?** Historical data from Beijing may not generalize to a different city or modern conditions; outputs are not official warnings.
40. **What would you improve next?** Validate on a newer external dataset, tune models, compare algorithms, and create additional explainability plots.
