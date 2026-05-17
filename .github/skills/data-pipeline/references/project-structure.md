# Project Structure Reference

## Source Tree

```
CTH621_Nhom3/
├── src/
│   ├── presentation/
│   │   └── run_pipeline.py          # CLI entry point (argparse/click)
│   ├── domain/
│   │   ├── eda.py                   # EDA + preprocessing logic
│   │   ├── classification.py        # Classification models
│   │   ├── regression.py            # Regression / time series models
│   │   └── clustering.py            # Clustering models
│   ├── data/
│   │   ├── loader.py                # Data loading (CSV, image, audio, video, text)
│   │   └── validator.py             # Input validation per group A/B/C
│   └── infrastructure/
│       ├── checkpoint.py            # progress.json tracking
│       └── logger.py                # Python logging setup
├── configs/
│   └── params.yaml                  # ALL hyperparameters + random_state
├── data/
│   ├── raw/                         # Original datasets (unmodified)
│   └── interim/                     # Intermediate .parquet files
├── outputs/
│   └── {dataset_name}/
│       ├── eda/
│       │   ├── raw/                 # Boxplots, null reports BEFORE processing
│       │   └── transformed/        # Plots + .xlsx/.csv AFTER processing
│       ├── ml/
│       │   ├── classification/
│       │   ├── regression/
│       │   └── clustering/
│       └── models/                  # .pkl / .joblib checkpoints
├── logs/
│   └── pipeline.log
├── progress.json
├── summary_results.csv
├── requirements.txt
├── environment.yml
└── ecosystem.config.js              # PM2 config for server deployment
```

## Naming Conventions
- Dataset names: `snake_case`, e.g. `student_performance`, `stock_prices`
- Python files: `snake_case`
- Config keys: `snake_case`
- Output folders mirror dataset names exactly
