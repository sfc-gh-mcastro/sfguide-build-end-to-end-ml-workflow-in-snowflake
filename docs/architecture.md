# Mortgage Lending MLOps - Architecture

> **Last Updated**: 2026-01-30

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Flow](#data-flow)
3. [Component Architecture](#component-architecture)
4. [Infrastructure](#infrastructure)
5. [Security & Access Control](#security--access-control)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              SNOWFLAKE PLATFORM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                     │
│   │  Raw Data   │─────►│   Feature   │─────►│  Training   │                     │
│   │   Table     │      │    Store    │      │   Dataset   │                     │
│   └─────────────┘      └─────────────┘      └─────────────┘                     │
│          │                    │                    │                             │
│          │                    │                    │                             │
│          ▼                    ▼                    ▼                             │
│   ┌─────────────────────────────────────────────────────────────────┐           │
│   │                    SNOWFLAKE NOTEBOOKS                           │           │
│   │   ┌─────────────────────────────────────────────────────────┐   │           │
│   │   │                                                          │   │           │
│   │   │   Feature Engineering ──► Model Training ──► Evaluation  │   │           │
│   │   │                                                          │   │           │
│   │   └─────────────────────────────────────────────────────────┘   │           │
│   └─────────────────────────────────────────────────────────────────┘           │
│          │                    │                    │                             │
│          ▼                    ▼                    ▼                             │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                     │
│   │  Experiment │      │    Model    │      │    Model    │                     │
│   │   Tracking  │      │   Registry  │      │   Monitor   │                     │
│   └─────────────┘      └─────────────┘      └─────────────┘                     │
│                               │                                                  │
│                               ▼                                                  │
│                        ┌─────────────┐                                          │
│                        │  Inference  │                                          │
│                        │   Service   │                                          │
│                        │   (SPCS)    │                                          │
│                        └─────────────┘                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

```yaml
Runtime:
  - Snowflake Notebooks (Container Runtime)
  - Compute Pool: CPU_X64_M

Python Dependencies:
  - snowflake-ml-python: 1.19.0
  - xgboost: latest
  - shap: latest
  - scikit-learn: latest
  - pandas: latest
  - numpy: latest

Snowflake Features:
  - Feature Store
  - Model Registry
  - Experiment Tracking
  - Model Monitor
  - Snowpark Container Services (SPCS)
```

---

## Data Flow

### Training Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING DATA FLOW                                  │
└──────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ MORTGAGE_LENDING│
    │   _DEMO_DATA    │
    │   (Raw Table)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐     ┌─────────────────┐
    │    Snowpark     │────►│  Feature Store  │
    │   Transforms    │     │  (Feature View) │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │   OneHotEncoder │────►│    Dataset      │
    │   (Categoricals)│     │   Generation    │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             ▼                       ▼
    ┌─────────────────┐     ┌─────────────────┐
    │   Train/Test    │     │   HPO Tuner     │
    │     Split       │────►│   (8+ trials)   │
    │   (70/30)       │     └────────┬────────┘
    └─────────────────┘              │
                                     ▼
                            ┌─────────────────┐
                            │  Model Registry │
                            │   (Versioned)   │
                            └─────────────────┘
```

### Feature Engineering Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING FLOW                               │
└──────────────────────────────────────────────────────────────────────────────┘

Raw Columns                    Transformations                    Features
─────────────                  ───────────────                    ────────

TS ─────────────────────────►  date_add(to_timestamp) ──────────► TIMESTAMP
                               month() ─────────────────────────► MONTH
                               dayofyear() ─────────────────────► DAY_OF_YEAR
                               dayofweek() ─────────────────────► DOTW

LOAN_AMOUNT_000s ───────────►  × 1000 ──────────────────────────► LOAN_AMOUNT

APPLICANT_INCOME_000s ──────►  × 1000 ──────────────────────────► INCOME
                               INCOME / LOAN_AMOUNT ────────────► INCOME_LOAN_RATIO
                               avg().over(county) ──────────────► MEAN_COUNTY_INCOME
                               INCOME > MEAN_COUNTY ────────────► HIGH_INCOME_FLAG

LOAN_AMOUNT + COUNTY ───────►  avg().over(30 days) ─────────────► AVG_THIRTY_DAY_LOAN

LOAN_PURPOSE_NAME ──────────►  OneHotEncoder ───────────────────► LOAN_PURPOSE_*

[Phase 2]
LOAN_TYPE_NAME ─────────────►  col() ───────────────────────────► LOAN_TYPE
COUNTY_NAME ────────────────►  avg(MORTGAGERESPONSE).over() ────► COUNTY_APPROVAL_RATE
                               LOAN / MEAN_COUNTY_INCOME ───────► LOAN_TO_COUNTY_RATIO
                               percent_rank().over(county) ─────► INCOME_PERCENTILE
LOAN_TYPE_NAME ─────────────►  avg(MORTGAGERESPONSE).over() ────► LOAN_TYPE_APPROVAL

[Phase 3]
APPLICANT_INCOME_000s ──────►  when(isNull(), 1) ───────────────► INCOME_MISSING_FLAG
```

### Inference Pipeline

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE DATA FLOW                                 │
└──────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  Input Request  │
                         │  (New Loan App) │
                         └────────┬────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              WAREHOUSE                                        │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │  Feature Store  │──►│  Model Version  │──►│   Prediction    │            │
│  │   (Lookup)      │   │   (Registry)    │   │   (0/1 + prob)  │            │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘            │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │   Explanation   │
                         │  (SHAP values)  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Output Result  │
                         │  + Audit Log    │
                         └─────────────────┘
```

---

## Component Architecture

### Feature Store

```
┌─────────────────────────────────────────────────────────────────┐
│                        FEATURE STORE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Entity: LOAN_ENTITY                                             │
│  ├── Join Key: LOAN_ID                                          │
│  └── Description: Features defined on a per loan level          │
│                                                                  │
│  Feature View: Mortgage_Feature_View                            │
│  ├── Version: VERSION_NUM (0, 1, 2, ...)                        │
│  ├── Timestamp Column: TIMESTAMP                                │
│  ├── Refresh Frequency: 1 day                                   │
│  └── Features:                                                  │
│      ├── MONTH                                                  │
│      ├── DAY_OF_YEAR                                            │
│      ├── DOTW                                                   │
│      ├── LOAN_AMOUNT                                            │
│      ├── INCOME                                                 │
│      ├── INCOME_LOAN_RATIO                                      │
│      ├── MEAN_COUNTY_INCOME                                     │
│      ├── HIGH_INCOME_FLAG                                       │
│      └── AVG_THIRTY_DAY_LOAN_AMOUNT                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Registry

```
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL REGISTRY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Model: MORTGAGE_LENDING_MLOPS_{VERSION_NUM}                    │
│  │                                                               │
│  ├── Version: XGB_BASE                                          │
│  │   ├── Algorithm: XGBoost                                     │
│  │   ├── Metrics: F1, Precision, Recall (Train/Test)            │
│  │   └── Alias: FIRST                                           │
│  │                                                               │
│  └── Version: XGB_OPTIMIZED                                     │
│      ├── Algorithm: XGBoost (HPO-tuned)                         │
│      ├── Metrics: F1, Precision, Recall (Train/Test)            │
│      ├── Alias: DEFAULT, LAST                                   │
│      └── Tag: PROD                                              │
│                                                                  │
│  Functions Available:                                            │
│  ├── PREDICT        - Binary prediction (0/1)                   │
│  ├── PREDICT_PROBA  - Probability scores                        │
│  └── EXPLAIN        - SHAP feature attributions                 │
│                                                                  │
│  Target Platforms:                                               │
│  ├── WAREHOUSE                                                  │
│  └── SNOWPARK_CONTAINER_SERVICES                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Experiment Tracking

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENT TRACKING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Experiment: E2E_MLOPS_HPO_Experiments                          │
│  │                                                               │
│  └── Runs (per HPO trial):                                      │
│      ├── Parameters Logged:                                     │
│      │   ├── max_depth                                          │
│      │   ├── n_estimators                                       │
│      │   ├── learning_rate                                      │
│      │   └── scale_pos_weight (Phase 4+)                        │
│      │                                                           │
│      └── Metrics Logged:                                        │
│          ├── F1_Train / F1_Test                                 │
│          ├── Precision_Train / Precision_Test                   │
│          └── Recall_Train / Recall_Test                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Monitor

```
┌─────────────────────────────────────────────────────────────────┐
│                       MODEL MONITOR                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Monitor: MORTGAGE_MODEL_MONITOR_{VERSION_NUM}                  │
│  │                                                               │
│  ├── Source Table: DEMO_MORTGAGE_LENDING_PREDICTION_LOG         │
│  │   ├── LOAN_ID (entity ID)                                    │
│  │   ├── PREDICTION (model output)                              │
│  │   ├── ACTUAL (ground truth)                                  │
│  │   └── TIMESTAMP                                              │
│  │                                                               │
│  ├── Model Version Refs:                                        │
│  │   ├── XGB_BASE                                               │
│  │   └── XGB_OPTIMIZED                                          │
│  │                                                               │
│  ├── Metrics Tracked:                                           │
│  │   ├── F1 Score                                               │
│  │   ├── Precision                                              │
│  │   ├── Recall                                                 │
│  │   └── Prediction Drift                                       │
│  │                                                               │
│  └── Refresh: Daily                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure

### Snowflake Objects

```
E2E_SNOW_MLOPS_DB (Database)
└── MLOPS_SCHEMA (Schema)
    │
    ├── Tables
    │   ├── MORTGAGE_LENDING_DEMO_DATA        # Raw source data
    │   ├── DEMO_MORTGAGE_LENDING_TRAIN_{N}   # Training data
    │   └── DEMO_MORTGAGE_LENDING_TEST_{N}    # Test data
    │
    ├── Feature Store
    │   ├── LOAN_ENTITY                        # Entity definition
    │   └── Mortgage_Feature_View              # Feature view
    │
    ├── Models
    │   ├── MORTGAGE_LENDING_MLOPS_0          # Version 0 (deprecated)
    │   └── MORTGAGE_LENDING_MLOPS_1          # Version 1 (current)
    │
    ├── Notebooks
    │   ├── TRAIN_DEPLOY_MONITOR_ML           # Original notebook
    │   └── TRAIN_DEPLOY_MONITOR_ML_V2        # Updated notebook
    │
    ├── Git Repository
    │   └── GITHUB_REPO_E2E_SNOW_MLOPS        # Git integration
    │
    └── Tags
        └── PROD                               # Production model tag
```

### Compute Resources

```sql
-- Compute Pool (Container Runtime)
CREATE COMPUTE POOL IF NOT EXISTS MLOPS_COMPUTE_POOL 
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_M;

-- Warehouse (SQL Execution)
CREATE OR REPLACE WAREHOUSE E2E_SNOW_MLOPS_WH 
  WITH WAREHOUSE_SIZE = 'MEDIUM';
```

### Notebook Configuration

| Setting | Value |
|---------|-------|
| Runtime | SYSTEM$BASIC_RUNTIME (Container) |
| Compute Pool | MLOPS_COMPUTE_POOL |
| Query Warehouse | E2E_SNOW_MLOPS_WH |
| Idle Timeout | 4 hours (14400 seconds) |

### Git Integration

```sql
-- API Integration
CREATE API INTEGRATION GITHUB_INTEGRATION_MCASTRO
  api_provider = git_https_api
  api_allowed_prefixes = ('https://github.com/sfc-gh-mcastro')
  enabled = true;

-- Git Repository
CREATE GIT REPOSITORY GITHUB_REPO_E2E_SNOW_MLOPS
  ORIGIN = 'https://github.com/sfc-gh-mcastro/sfguide-build-end-to-end-ml-workflow-in-snowflake'
  API_INTEGRATION = 'GITHUB_INTEGRATION_MCASTRO';
```

---

## Security & Access Control

### Role Hierarchy

```
ACCOUNTADMIN
    │
    └── E2E_SNOW_MLOPS_ROLE
            │
            ├── CREATE DATABASE
            ├── CREATE COMPUTE POOL
            ├── CREATE WAREHOUSE
            ├── BIND SERVICE ENDPOINT
            └── CREATE INTEGRATION
```

### Permissions

```sql
-- Role creation and grants
USE ROLE ACCOUNTADMIN;

CREATE OR REPLACE ROLE E2E_SNOW_MLOPS_ROLE;

GRANT CREATE DATABASE ON ACCOUNT TO ROLE E2E_SNOW_MLOPS_ROLE;
GRANT CREATE COMPUTE POOL ON ACCOUNT TO ROLE E2E_SNOW_MLOPS_ROLE;
GRANT CREATE WAREHOUSE ON ACCOUNT TO ROLE E2E_SNOW_MLOPS_ROLE;
GRANT BIND SERVICE ENDPOINT ON ACCOUNT TO ROLE E2E_SNOW_MLOPS_ROLE;
GRANT CREATE INTEGRATION ON ACCOUNT TO ROLE E2E_SNOW_MLOPS_ROLE;

-- Grant role to user
GRANT ROLE E2E_SNOW_MLOPS_ROLE TO USER <username>;
```

---

## Related Documents

- [Project Spec](project_spec.md) - Requirements and API specs
- [Project Status](project_status.md) - Current progress
- [Changelog](changelog.md) - Version history

---

*Document maintained with Cortex Code*
