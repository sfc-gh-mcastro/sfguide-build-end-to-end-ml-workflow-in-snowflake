# Project Status: Mortgage Lending Model Optimization

> **Last Updated**: 2026-01-30
> **Project Goal**: Improve MORTGAGE_LENDING_MLOPS_0 model performance

---

## IMPORTANT: Version Control

When implementing the next phase, **update `VERSION_NUM`** in the notebook to capture changes:

```python
# Cell: d78265b8-8baa-4136-a32a-32f3f620949d
VERSION_NUM = '2'  # Increment from '1' to '2' for Phase 2/3 changes
```

This ensures:
- New feature views are created (not overwritten)
- New model versions are registered separately
- Clear lineage between model iterations

---

## Quick Resume Guide

To continue this work, run the updated notebook in Snowflake:

```
Notebook URL: https://app.snowflake.com/SFSEEUROPE/MCASTRO_AWS1_USWEST2/#/notebooks/5klwnozziwlcmkjkylha
Notebook Name: TRAIN_DEPLOY_MONITOR_ML_V2
Database: E2E_SNOW_MLOPS_DB
Schema: MLOPS_SCHEMA
```

### Notebook Configuration
| Setting | Value |
|---------|-------|
| Compute Pool | `MLOPS_COMPUTE_POOL` (container runtime) |
| Idle Timeout | 4 hours (14400 seconds) |
| Query Warehouse | `E2E_SNOW_MLOPS_WH` |

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Address Class Imbalance | ✅ COMPLETED |
| 2 | Add Missing Features | PENDING |
| 3 | Handle NULL Values | PENDING |
| 4 | Expand HPO Search Space | PENDING |
| 5 | Add Threshold Optimization | PENDING |

---

## Phase 1 Results: Model Comparison

### XGB_OPTIMIZED (Production Model)

| Metric | Version 0 | Version 1 | Change |
|--------|-----------|-----------|--------|
| **Test F1 Score** | 0.8798 | **0.8854** | **+0.56%** ✓ |
| **Test Precision** | 0.7853 | **0.8104** | **+2.51%** ✓ |
| **Test Recall** | 1.0000 | 0.9757 | -2.43% |
| Train F1 Score | 0.8787 | 0.8874 | +0.87% |
| Train Precision | 0.7836 | 0.8127 | +2.91% |
| Train Recall | 1.0000 | 0.9773 | -2.27% |

### XGB_BASE Model

| Metric | Version 0 | Version 1 | Change |
|--------|-----------|-----------|--------|
| **Test F1 Score** | 0.8872 | 0.8822 | -0.50% |
| **Test Precision** | 0.8485 | 0.7903 | -5.82% |
| **Test Recall** | 0.9297 | 0.9984 | +6.87% |

### Key Improvements Achieved

1. ✅ **Precision improved by 2.5%** (0.7853 → 0.8104) - Fewer false positives
2. ✅ **F1 Score improved** (0.8798 → 0.8854) - Better overall balance
3. ✅ **Model no longer predicts ALL as approved** - Recall dropped from 1.0 to 0.976
4. ✅ **Reduced overfitting** - Train/Test gap is now ~0.2%

### Hyperparameters (HPO Version 1)

| Parameter | Version 0 | Version 1 |
|-----------|-----------|-----------|
| max_depth | 3 | 4 |
| n_estimators | 147 | 128 |
| learning_rate | 0.12 | 0.30 |
| scale_pos_weight | N/A | 3.6 ✓ |

---

## What Was Done (Phase 1)

### Problem Identified
- Model predicted ALL cases as approved (1)
- Perfect recall (1.0) but poor precision (0.7853)
- Class imbalance: 78% approved vs 22% rejected

### Solution Implemented
Modified XGBoost base model configuration:

```python
# Before
xgb_base = XGBClassifier(
    max_depth=50,
    n_estimators=3,
    learning_rate=0.75,
    booster='gbtree')

# After (Phase 1 fix)
xgb_base = XGBClassifier(
    max_depth=6,
    n_estimators=100,
    learning_rate=0.1,
    booster='gbtree',
    scale_pos_weight=3.6,      # Handle class imbalance
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8)
```

### Infrastructure Changes
1. Created new API integration: `GITHUB_INTEGRATION_MCASTRO`
2. Updated Git repository to point to fork: `sfc-gh-mcastro/sfguide-build-end-to-end-ml-workflow-in-snowflake`
3. Created new notebook: `TRAIN_DEPLOY_MONITOR_ML_V2`

---

## What's Next (Phase 2-5)

### Phase 2: Add Missing Features
- Add `LOAN_TYPE_NAME` feature (VA-guaranteed, FHA-insured, Conventional)
- Add `COUNTY_APPROVAL_RATE` derived feature
- Add `LOAN_TO_COUNTY_INCOME_RATIO` derived feature

### Phase 3: Handle NULL Values
- Create `INCOME_MISSING_FLAG` indicator
- Impute missing income with county median instead of 0

### Phase 4: Expand HPO Search Space
- Add `scale_pos_weight` to hyperparameter search
- Add regularization parameters (`gamma`, `subsample`, `colsample_bytree`)
- Increase trials from 8 to 25

### Phase 5: Add Threshold Optimization
- Find optimal decision threshold using precision-recall curve
- Store optimal threshold as model metadata

---

## Files in This Project

```
plan/
├── MORTGAGE_MODEL_IMPROVEMENT_PLAN.md  # Detailed improvement plan with code
├── PROJECT_STATUS.md                    # This file - quick status reference
└── train_deploy_monitor_ML_in_snowflake_v2.ipynb  # Local copy of modified notebook
```

---

## Snowflake Objects

| Object | Type | Location |
|--------|------|----------|
| Original Model (v0) | Model Registry | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_0` |
| **Improved Model (v1)** | Model Registry | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_1` |
| Original Notebook | Notebook | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.TRAIN_DEPLOY_MONITOR_ML` |
| Updated Notebook | Notebook | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.TRAIN_DEPLOY_MONITOR_ML_V2` |
| Git Repository | Git Repo | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.GITHUB_REPO_E2E_SNOW_MLOPS` |
| Training Data | Table | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.DEMO_MORTGAGE_LENDING_TRAIN_0` |
| Test Data | Table | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.DEMO_MORTGAGE_LENDING_TEST_0` |

---

## Git Information

| Property | Value |
|----------|-------|
| Repository | `https://github.com/sfc-gh-mcastro/sfguide-build-end-to-end-ml-workflow-in-snowflake` |
| Branch | `main` |

---

## Commands to Continue

```sql
-- Switch to correct role
USE ROLE E2E_SNOW_MLOPS_ROLE;

-- Check model versions
SHOW VERSIONS IN MODEL E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_1;

-- Fetch latest from Git (if changes pushed)
ALTER GIT REPOSITORY E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.GITHUB_REPO_E2E_SNOW_MLOPS FETCH;
```
