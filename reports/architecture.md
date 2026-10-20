# AirLens architecture

```mermaid
flowchart TD
    A[UCI PRSA ZIP / station CSVs] --> B[Extract and validate]
    B --> C[Preprocess timestamps, duplicates, invalid readings]
    C --> D[Cleaned observations CSV]
    C --> E[SQLite warehouse: dimensions and fact table]
    D --> F[Exploratory analysis and charts]
    D --> G[Apriori association rules]
    D --> H[K-Means pollution regimes]
    D --> I[Chronological next-hour models]
    F --> J[Outputs and insights]
    G --> J
    H --> J
    I --> J
    E --> K[Streamlit dashboard]
    J --> K
```

The numerical imputer and categorical encoder for predictive models are fitted within model pipelines, using training rows only. The model train/test split is ordered by target timestamp.
