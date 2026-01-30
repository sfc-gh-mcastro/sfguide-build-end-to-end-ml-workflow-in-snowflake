# Project Status: Mortgage Lending Model Optimization

> **Last Updated**: 2026-01-29 12:50 PST
> **Project Goal**: Improve MORTGAGE_LENDING_MLOPS_0 model performance

---

## IMPORTANT: Version Control

When implementing the next phase, **update `VERSION_NUM`** in the notebook to capture changes:

```python
# Cell: d78265b8-8baa-4136-a32a-32f3f620949d
VERSION_NUM = '1'  # Increment from '0' to '1' for Phase 2 changes
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

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Address Class Imbalance | COMPLETED |
| 2 | Add Missing Features | PENDING |
| 3 | Handle NULL Values | PENDING |
| 4 | Expand HPO Search Space | PENDING |
| 5 | Add Threshold Optimization | PENDING |

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
| Original Model | Model Registry | `E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_0` |
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
| Latest Commit | `88f15ba` - "fix: Address class imbalance in XGBoost base model (Phase 1)" |

---

## Expected Results After All Phases

| Metric | Before | After (Expected) |
|--------|--------|------------------|
| Precision | 0.7853 | 0.88 - 0.93 |
| Recall | 1.0000 | 0.90 - 0.95 |
| F1 Score | 0.8798 | 0.91 - 0.95 |

---

## Commands to Continue

```sql
-- Switch to correct role
USE ROLE E2E_SNOW_MLOPS_ROLE;

-- Check model status
SHOW VERSIONS IN MODEL E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_0;

-- Fetch latest from Git (if changes pushed)
ALTER GIT REPOSITORY E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.GITHUB_REPO_E2E_SNOW_MLOPS FETCH;
```
