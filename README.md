# End-to-End ML Workflow in Snowflake

> **Mortgage Lending Prediction Model** - Demonstrating Snowflake's ML Platform

[![Snowflake](https://img.shields.io/badge/Snowflake-ML%20Platform-29B5E8)](https://www.snowflake.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

This project demonstrates a complete ML workflow in Snowflake for predicting mortgage loan approval likelihood, featuring:

- **Feature Store** - Track and version engineered features
- **Model Training** - XGBoost with distributed hyperparameter optimization
- **Model Registry** - Version control, metadata tracking, and inference
- **Model Monitoring** - Production performance tracking and drift detection
- **Explainability** - SHAP-based feature attributions

## Project Structure

```
├── data/                          # Source data files
│   └── MORTGAGE_LENDING_DEMO_DATA.csv.zip
│
├── notebooks/                     # Jupyter notebooks
│   ├── train_deploy_monitor_ML_in_snowflake.ipynb  # Main notebook
│   └── archive/                   # Previous versions
│
├── sql/                           # SQL scripts
│   └── setup.sql                  # Infrastructure setup
│
├── docs/                          # Documentation
│   ├── project_spec.md            # Requirements & API specs
│   ├── architecture.md            # System design
│   ├── project_status.md          # Current progress
│   └── changelog.md               # Version history
│
├── assets/                        # Images and screenshots
│
├── environment.yml                # Python dependencies
├── LICENSE
└── README.md
```

## Quick Start

### Prerequisites

- Snowflake account with ML features enabled
- `ACCOUNTADMIN` role (for initial setup)

### Setup

1. **Run infrastructure setup**
   ```sql
   -- Execute sql/setup.sql in Snowflake
   ```

2. **Open the notebook**
   ```
   Database: E2E_SNOW_MLOPS_DB
   Schema: MLOPS_SCHEMA
   Notebook: TRAIN_DEPLOY_MONITOR_ML
   ```

3. **Run all cells** to:
   - Load and engineer features
   - Train baseline and optimized models
   - Register models and set up monitoring

## Features

### 1. Feature Store
- Store feature definitions for reproducible ML features
- Version control for feature views
- Point-in-time correct feature retrieval

### 2. Model Training
- **Baseline XGBoost**: Quick baseline with regularization
- **Optimized XGBoost**: Distributed HPO with experiment tracking
- Class imbalance handling via `scale_pos_weight`

### 3. Model Registry
- Version and tag models (BASE, OPTIMIZED, PROD)
- Automatic lineage tracking
- Built-in inference functions

### 4. Model Monitoring
- Track F1, Precision, Recall over time
- Detect prediction drift
- Compare model versions side-by-side

### 5. Explainability
- SHAP-based feature attributions
- Built-in visualization tools
- Per-prediction explanations

## Current Performance

| Metric | Baseline (v0) | Improved (v1) |
|--------|---------------|---------------|
| Test F1 | 0.8798 | **0.8854** |
| Test Precision | 0.7853 | **0.8104** |
| Test Recall | 1.0000 | 0.9757 |

## Documentation

| Document | Description |
|----------|-------------|
| [Project Spec](docs/project_spec.md) | Full requirements, API specs, technical details |
| [Architecture](docs/architecture.md) | System design and data flow diagrams |
| [Project Status](docs/project_status.md) | Current progress and next steps |
| [Changelog](docs/changelog.md) | Version history |

## Resources

- [Snowflake ML Documentation](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview)
- [QuickStart Guide](https://quickstarts.snowflake.com/guide/end-to-end-ml-workflow)
- [Feature Store Guide](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)
- [Model Registry Guide](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Maintained with Cortex Code*
