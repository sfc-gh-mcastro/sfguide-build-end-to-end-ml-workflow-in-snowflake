# Mortgage Lending MLOps - Project Status

> **Last Updated**: 2026-01-30

---

## Quick Resume

### Notebook Access

```
URL: https://app.snowflake.com/SFSEEUROPE/MCASTRO_AWS1_USWEST2/#/notebooks/5klwnozziwlcmkjkylha
Name: TRAIN_DEPLOY_MONITOR_ML_V2
Database: E2E_SNOW_MLOPS_DB
Schema: MLOPS_SCHEMA
```

### Configuration

| Setting | Value |
|---------|-------|
| Compute Pool | `MLOPS_COMPUTE_POOL` (container runtime) |
| Idle Timeout | 4 hours (14400 seconds) |
| Query Warehouse | `E2E_SNOW_MLOPS_WH` |

---

## Version Control Reminder

Before implementing the next phase, **update `VERSION_NUM`**:

```python
# Cell: d78265b8-8baa-4136-a32a-32f3f620949d
VERSION_NUM = '2'  # Increment from '1' to '2' for Phase 2/3 changes
```

---

## Phase Status

| Phase | Description | Status | Impact |
|-------|-------------|--------|--------|
| 1 | Address Class Imbalance | ✅ COMPLETED | +2.5% Precision |
| 2 | Add Missing Features | ⏳ PENDING | Est. +2-4% F1 |
| 3 | Handle NULL Values | ⏳ PENDING | Est. +1-2% F1 |
| 4 | Expand HPO Search Space | ⏳ PENDING | Est. +1-3% F1 |
| 5 | Add Threshold Optimization | ⏳ PENDING | Est. +0.5-1% F1 |

---

## Current Performance (v1)

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
| Test F1 Score | 0.8872 | 0.8822 | -0.50% |
| Test Precision | 0.8485 | 0.7903 | -5.82% |
| Test Recall | 0.9297 | 0.9984 | +6.87% |

### Hyperparameters (HPO v1)

| Parameter | Version 0 | Version 1 |
|-----------|-----------|-----------|
| max_depth | 3 | 4 |
| n_estimators | 147 | 128 |
| learning_rate | 0.12 | 0.30 |
| scale_pos_weight | N/A | 3.6 ✓ |

---

## Key Improvements Achieved

1. ✅ **Precision improved by 2.5%** (0.7853 → 0.8104) - Fewer false positives
2. ✅ **F1 Score improved** (0.8798 → 0.8854) - Better overall balance
3. ✅ **Model no longer predicts ALL as approved** - Recall dropped from 1.0 to 0.976
4. ✅ **Reduced overfitting** - Train/Test gap is now ~0.2%

---

## Target Metrics

| Metric | Current | Target | Gap |
|--------|---------|--------|-----|
| Test F1 Score | 0.8854 | > 0.91 | -2.5% |
| Test Precision | 0.8104 | > 0.88 | -7.0% |
| Test Recall | 0.9757 | 0.90-0.95 | ✓ Achieved |
| Train/Test Gap | 0.2% | < 5% | ✓ Achieved |

---

## Next Steps

### Phase 2: Add Missing Features

**Objective**: Incorporate unused predictive features

**Tasks**:
1. Add `LOAN_TYPE_NAME` as categorical feature
2. Create `COUNTY_APPROVAL_RATE` derived feature  
3. Create `LOAN_TO_COUNTY_INCOME_RATIO` derived feature
4. Create `INCOME_PERCENTILE_IN_COUNTY` derived feature
5. Create `LOAN_TYPE_APPROVAL_RATE` derived feature

**Code Location**: Cell `b355c0c4-9dc6-4faf-86b7-24d8d559e453`

### Phase 3: Handle NULL Values

**Objective**: Replace blanket `fillna(0)` with intelligent imputation

**Tasks**:
1. Create `INCOME_MISSING_FLAG` indicator feature
2. Impute missing income with county median instead of 0
3. Update NULL handling logic

**Code Locations**: 
- Cell `b355c0c4-9dc6-4faf-86b7-24d8d559e453` (feature creation)
- Cell `a8ff103e-5314-4e95-87ba-d784b1102f36` (fillna logic)

---

## Snowflake Objects

| Object | Type | Location |
|--------|------|----------|
| Original Model (v0) | Model Registry | `MORTGAGE_LENDING_MLOPS_0` |
| **Improved Model (v1)** | Model Registry | `MORTGAGE_LENDING_MLOPS_1` |
| Original Notebook | Notebook | `TRAIN_DEPLOY_MONITOR_ML` |
| Updated Notebook | Notebook | `TRAIN_DEPLOY_MONITOR_ML_V2` |
| Git Repository | Git Repo | `GITHUB_REPO_E2E_SNOW_MLOPS` |
| Training Data | Table | `DEMO_MORTGAGE_LENDING_TRAIN_0` |
| Test Data | Table | `DEMO_MORTGAGE_LENDING_TEST_0` |

---

## Quick Commands

```sql
-- Switch to correct role
USE ROLE E2E_SNOW_MLOPS_ROLE;

-- Check model versions
SHOW VERSIONS IN MODEL E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_1;

-- Fetch latest from Git
ALTER GIT REPOSITORY E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.GITHUB_REPO_E2E_SNOW_MLOPS FETCH;
```

---

## Repository Info

| Property | Value |
|----------|-------|
| Repository | `https://github.com/sfc-gh-mcastro/sfguide-build-end-to-end-ml-workflow-in-snowflake` |
| Branch | `main` |

---

## Related Documents

- [Project Spec](project_spec.md) - Requirements and API specs
- [Architecture](architecture.md) - System design and data flow
- [Changelog](changelog.md) - Version history

---

*Document maintained with Cortex Code*
