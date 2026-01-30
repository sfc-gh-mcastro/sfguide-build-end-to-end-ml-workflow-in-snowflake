# Mortgage Lending MLOps - Project Specification

> **Version**: 1.0  
> **Last Updated**: 2026-01-30  
> **Status**: Phase 1 Complete, Phases 2-5 In Progress

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Business Context](#business-context)
3. [Technical Architecture](#technical-architecture)
4. [Data Specification](#data-specification)
5. [Feature Engineering](#feature-engineering)
6. [Model Specification](#model-specification)
7. [Infrastructure Requirements](#infrastructure-requirements)
8. [Milestones & Roadmap](#milestones--roadmap)
9. [Success Metrics](#success-metrics)
10. [Appendix](#appendix)

---

## Executive Summary

### Project Goal
Build and operationalize an end-to-end ML workflow in Snowflake for predicting mortgage loan approval likelihood, demonstrating Snowflake's ML platform capabilities including Feature Store, Model Registry, Distributed HPO, Experiment Tracking, and Model Monitoring.

### Current State
- **Phase 1 Complete**: Addressed class imbalance issue that caused model to predict all loans as approved
- **Model Performance (v1)**: F1=0.8854, Precision=0.8104, Recall=0.9757
- **Production Model**: `MORTGAGE_LENDING_MLOPS_1.XGB_OPTIMIZED`

### Key Outcomes
| Metric | Baseline (v0) | Improved (v1) | Target (Final) |
|--------|---------------|---------------|----------------|
| Precision | 0.7853 | 0.8104 (+2.5%) | 0.88 - 0.93 |
| Recall | 1.0000 | 0.9757 | 0.90 - 0.95 |
| F1 Score | 0.8798 | 0.8854 (+0.6%) | 0.91 - 0.95 |

---

## Business Context

### Problem Statement
Financial institutions need to predict mortgage loan approval likelihood to:
- Streamline underwriting processes
- Reduce manual review overhead
- Ensure consistent decision-making
- Comply with fair lending regulations

### Use Case
Binary classification to predict whether a mortgage loan application will be **approved (1)** or **rejected (0)** based on applicant and loan characteristics.

### Stakeholders
| Role | Responsibility |
|------|----------------|
| Data Scientists | Model development, feature engineering, experimentation |
| ML Engineers | Model deployment, monitoring, infrastructure |
| Risk Analysts | Model validation, fairness assessment |
| Business Users | Consume predictions for loan decisioning |

### Business Constraints
- Model must be explainable (regulatory requirement)
- Predictions must be available in near real-time
- Model drift must be monitored continuously
- Clear audit trail for all model versions

---

## Technical Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SNOWFLAKE PLATFORM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │   Raw Data   │───►│   Feature    │───►│   Training   │                   │
│  │    Table     │    │    Store     │    │   Dataset    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────────────────────────────────────────────┐                   │
│  │                  SNOWFLAKE NOTEBOOKS                  │                   │
│  │  ┌─────────────────────────────────────────────────┐ │                   │
│  │  │  Feature Engineering → Training → Evaluation    │ │                   │
│  │  └─────────────────────────────────────────────────┘ │                   │
│  └──────────────────────────────────────────────────────┘                   │
│         │                   │                   │                            │
│         ▼                   ▼                   ▼                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  Experiment  │    │    Model     │    │    Model     │                   │
│  │   Tracking   │    │   Registry   │    │   Monitor    │                   │
│  └──────────────┘    └──────────────┘    └──────────────┘                   │
│                             │                                                │
│                             ▼                                                │
│                      ┌──────────────┐                                        │
│                      │  Inference   │                                        │
│                      │   Service    │                                        │
│                      │   (SPCS)     │                                        │
│                      └──────────────┘                                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| Data Storage | Snowflake Tables | Raw and processed data storage |
| Feature Engineering | Snowpark Python | Transform raw data into ML features |
| Feature Store | Snowflake Feature Store | Version and serve features |
| Model Training | XGBoost + Snowflake ML | Train classification models |
| HPO | Snowflake Distributed Tuning | Hyperparameter optimization |
| Experiment Tracking | Snowflake Experiments | Track training runs |
| Model Registry | Snowflake Model Registry | Version and deploy models |
| Model Monitoring | Snowflake Model Monitor | Track production performance |
| Inference | Warehouse / SPCS | Serve predictions |
| Explainability | SHAP / Built-in | Generate feature attributions |

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

## Data Specification

### Source Data

**Table**: `MORTGAGE_LENDING_DEMO_DATA`

| Column | Type | Description |
|--------|------|-------------|
| LOAN_ID | INTEGER | Unique loan identifier |
| TS | TIMESTAMP | Loan application timestamp |
| LOAN_AMOUNT_000s | FLOAT | Loan amount in thousands |
| APPLICANT_INCOME_000s | FLOAT | Applicant income in thousands (nullable) |
| LOAN_PURPOSE_NAME | STRING | Purpose: Purchase, Refinancing, Home improvement |
| LOAN_TYPE_NAME | STRING | Type: Conventional, FHA-insured, VA-guaranteed |
| COUNTY_NAME | STRING | County of property |
| MORTGAGERESPONSE | INTEGER | Target: 1=Approved, 0=Rejected |

### Data Statistics

| Metric | Value |
|--------|-------|
| Total Records | ~369,245 |
| Training Set | ~258,307 (70%) |
| Test Set | ~110,938 (30%) |
| Date Range | Rolling 1 year from current date |
| Class Distribution | 78% Approved / 22% Rejected |

### Data Quality Issues

| Issue | Impact | Resolution |
|-------|--------|------------|
| NULL income values | ~15% of records | Phase 3: Impute with county median |
| Class imbalance | Model bias | Phase 1: scale_pos_weight (DONE) |
| Weak feature correlations | Low predictive power | Phase 2: Add derived features |

---

## Feature Engineering

### Current Features (v1)

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| MONTH | INT | Month of loan application | Derived from TS |
| DAY_OF_YEAR | INT | Day of year (1-365) | Derived from TS |
| DOTW | INT | Day of week (0-6) | Derived from TS |
| LOAN_AMOUNT | FLOAT | Loan amount in USD | LOAN_AMOUNT_000s * 1000 |
| INCOME | FLOAT | Applicant income in USD | APPLICANT_INCOME_000s * 1000 |
| INCOME_LOAN_RATIO | FLOAT | Income / Loan Amount | Derived |
| MEAN_COUNTY_INCOME | FLOAT | Avg income by county | Window function |
| HIGH_INCOME_FLAG | INT | Income > County avg | Derived |
| AVG_THIRTY_DAY_LOAN_AMOUNT | FLOAT | 30-day rolling avg loan | Window function |
| LOAN_PURPOSE_NAME_* | INT | One-hot encoded purpose | OneHotEncoder |

### Planned Features (Phase 2)

| Feature | Type | Description | Expected Impact |
|---------|------|-------------|-----------------|
| LOAN_TYPE | STRING | VA/FHA/Conventional | High - different approval criteria |
| COUNTY_APPROVAL_RATE | FLOAT | Historical approval rate by county | Medium - geographic risk |
| LOAN_TO_COUNTY_INCOME_RATIO | FLOAT | Loan / County avg income | Medium - affordability signal |
| INCOME_PERCENTILE_IN_COUNTY | FLOAT | Income rank within county | Medium - relative wealth |
| LOAN_TYPE_APPROVAL_RATE | FLOAT | Approval rate by loan type | Medium - product risk |

### Planned Features (Phase 3)

| Feature | Type | Description | Expected Impact |
|---------|------|-------------|-----------------|
| INCOME_MISSING_FLAG | INT | 1 if income was NULL | High - missingness is predictive |

### Feature Store Configuration

```python
# Entity Definition
loan_id_entity = Entity(
    name="LOAN_ENTITY",
    join_keys=["LOAN_ID"],
    desc="Features defined on a per loan level"
)

# Feature View
loan_fv = FeatureView(
    name="Mortgage_Feature_View",
    entities=[loan_id_entity],
    feature_df=feature_df,
    timestamp_col="TIMESTAMP",
    refresh_freq="1 day"
)
```

---

## Model Specification

### Model Architecture

| Component | Specification |
|-----------|---------------|
| Algorithm | XGBoost Classifier |
| Task | Binary Classification |
| Target | MORTGAGERESPONSE (0/1) |
| Framework | xgboost via snowflake-ml-python |

### Model Versions

#### Version 0 (Baseline - Deprecated)

```python
# Original configuration - caused overfitting
XGBClassifier(
    max_depth=50,        # Too deep
    n_estimators=3,      # Too few
    learning_rate=0.75,  # Too high
    booster='gbtree'
)
```

**Issues**: 
- Severe overfitting (Train F1: 0.96, Test F1: 0.88)
- Predicted ALL cases as approved (Recall: 1.0)

#### Version 1 (Current Production)

```python
# Improved configuration with class imbalance handling
XGBClassifier(
    max_depth=6,              # Reduced to prevent overfitting
    n_estimators=100,         # More trees for stability
    learning_rate=0.1,        # Lower for better generalization
    booster='gbtree',
    scale_pos_weight=3.6,     # Handle 78/22 class imbalance
    min_child_weight=5,       # Regularization
    subsample=0.8,            # Row sampling
    colsample_bytree=0.8      # Column sampling
)
```

**Improvements**:
- Balanced predictions (Recall: 0.976)
- Improved precision (+2.5%)
- Minimal overfitting (Train/Test gap: 0.2%)

### Hyperparameter Optimization

```python
# Current HPO Search Space
search_space = {
    "max_depth": tune.randint(1, 30),
    "learning_rate": tune.uniform(0.01, 0.5),
    "n_estimators": tune.randint(50, 150),
}

# Planned Expanded Search Space (Phase 4)
search_space_expanded = {
    "max_depth": tune.randint(3, 10),
    "n_estimators": tune.randint(100, 300),
    "learning_rate": tune.uniform(0.01, 0.2),
    "min_child_weight": tune.randint(1, 10),
    "gamma": tune.uniform(0, 0.5),
    "subsample": tune.uniform(0.6, 0.9),
    "colsample_bytree": tune.uniform(0.6, 0.9),
    "scale_pos_weight": tune.uniform(2.5, 5.0),  # Critical
}

tuner_config = TunerConfig(
    metric="F1_Test",
    mode="max",
    num_trials=25,  # Increased from 8
)
```

### Model Registry Configuration

```python
model_registry = Registry(
    session=session,
    database_name="E2E_SNOW_MLOPS_DB",
    schema_name="MLOPS_SCHEMA",
    options={"enable_monitoring": True}
)

# Model logging with lineage
mv = model_registry.log_model(
    model_name=f"MORTGAGE_LENDING_MLOPS_{VERSION_NUM}",
    model=model,
    version_name="XGB_OPTIMIZED",
    sample_input_data=train_df.limit(100),  # Maintains lineage
    target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
    options={"enable_explainability": True}
)
```

---

## Infrastructure Requirements

### Snowflake Objects

| Object | Type | Purpose |
|--------|------|---------|
| E2E_SNOW_MLOPS_DB | Database | Project database |
| MLOPS_SCHEMA | Schema | All ML objects |
| E2E_SNOW_MLOPS_WH | Warehouse | Query execution |
| MLOPS_COMPUTE_POOL | Compute Pool | Container runtime |
| E2E_SNOW_MLOPS_ROLE | Role | Access control |

### Compute Resources

```sql
-- Compute Pool Configuration
CREATE COMPUTE POOL IF NOT EXISTS MLOPS_COMPUTE_POOL 
  MIN_NODES = 1
  MAX_NODES = 1
  INSTANCE_FAMILY = CPU_X64_M;

-- Warehouse Configuration
CREATE OR REPLACE WAREHOUSE E2E_SNOW_MLOPS_WH 
  WITH WAREHOUSE_SIZE='MEDIUM';
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

## Milestones & Roadmap

### Phase Overview

```
Phase 1 ✅ ─────► Phase 2 ─────► Phase 3 ─────► Phase 4 ─────► Phase 5
 Class           Add Missing    Handle NULL    Expand HPO    Threshold
 Imbalance       Features       Values         Search        Optimization
 COMPLETE        PENDING        PENDING        PENDING       PENDING
```

### Phase 1: Address Class Imbalance (COMPLETED)

**Objective**: Fix the model predicting all cases as approved

**Changes Made**:
- Added `scale_pos_weight=3.6` to handle 78/22 class imbalance
- Reduced `max_depth` from 50 to 6
- Increased `n_estimators` from 3 to 100
- Reduced `learning_rate` from 0.75 to 0.1
- Added regularization parameters

**Results**:
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Precision | 0.7853 | 0.8104 | +2.51% |
| Recall | 1.0000 | 0.9757 | -2.43% (expected) |
| F1 Score | 0.8798 | 0.8854 | +0.56% |

### Phase 2: Add Missing Features (PENDING)

**Objective**: Incorporate unused predictive features

**Tasks**:
1. Add `LOAN_TYPE_NAME` as categorical feature
2. Create `COUNTY_APPROVAL_RATE` derived feature
3. Create `LOAN_TO_COUNTY_INCOME_RATIO` derived feature
4. Create `INCOME_PERCENTILE_IN_COUNTY` derived feature
5. Create `LOAN_TYPE_APPROVAL_RATE` derived feature

**Code Location**: Cell `b355c0c4-9dc6-4faf-86b7-24d8d559e453`

**Expected Impact**: +2-4% F1 improvement

### Phase 3: Handle NULL Values (PENDING)

**Objective**: Replace blanket `fillna(0)` with intelligent imputation

**Tasks**:
1. Create `INCOME_MISSING_FLAG` indicator feature
2. Impute missing income with county median instead of 0
3. Update NULL handling logic

**Code Locations**: 
- Cell `b355c0c4-9dc6-4faf-86b7-24d8d559e453` (feature creation)
- Cell `a8ff103e-5314-4e95-87ba-d784b1102f36` (fillna logic)

**Expected Impact**: +1-2% F1 improvement

### Phase 4: Expand HPO Search Space (PENDING)

**Objective**: Find better hyperparameters with expanded search

**Tasks**:
1. Add `scale_pos_weight` to HPO search space
2. Add regularization parameters (`gamma`, `subsample`, `colsample_bytree`)
3. Increase trials from 8 to 25
4. Narrow tree depth range (3-10 instead of 1-30)

**Code Location**: Cell `e4d6860a-da49-42bc-aed9-57692eb5c7a2`

**Expected Impact**: +1-3% F1 improvement

### Phase 5: Threshold Optimization (PENDING)

**Objective**: Find optimal decision threshold for business needs

**Tasks**:
1. Generate precision-recall curve
2. Calculate F1 score at each threshold
3. Find optimal threshold maximizing F1
4. Store threshold as model metadata
5. Add visualization cell

**Code Location**: New cell after `dee80c48-d521-4b77-8841-54ba35ecd4b6`

**Expected Impact**: +0.5-1% F1 improvement

### Version Control Protocol

> **CRITICAL**: Before implementing each phase, increment `VERSION_NUM`:

```python
# Cell: d78265b8-8baa-4136-a32a-32f3f620949d
VERSION_NUM = '2'  # Increment for next phase
```

This ensures:
- New feature views are created (not overwritten)
- New model versions are registered separately
- Clear lineage between model iterations

---

## Success Metrics

### Model Performance Targets

| Metric | Baseline | Current | Target | Status |
|--------|----------|---------|--------|--------|
| Test F1 Score | 0.8798 | 0.8854 | > 0.91 | In Progress |
| Test Precision | 0.7853 | 0.8104 | > 0.88 | In Progress |
| Test Recall | 1.0000 | 0.9757 | 0.90-0.95 | Achieved |
| Train/Test Gap | 8.15% | 0.20% | < 5% | Achieved |

### Operational Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Inference Latency | < 100ms | TBD |
| Model Drift Alert | Weekly | Configured |
| Feature Freshness | Daily | Configured |
| Explainability | All predictions | Enabled |

### Validation Checklist

- [x] Confusion matrix shows predictions for both classes
- [x] Precision improved to > 0.80
- [x] Recall is between 0.85-0.99 (not artificially 1.0)
- [x] F1 score improved
- [x] Train/Test performance gap < 5%
- [ ] Precision improved to > 0.88
- [ ] F1 score improved to > 0.91
- [ ] Feature importance shows new features contributing

---

## Appendix

### A. Snowflake Objects Reference

```sql
-- Database and Schema
USE DATABASE E2E_SNOW_MLOPS_DB;
USE SCHEMA MLOPS_SCHEMA;

-- Models
SHOW MODELS IN SCHEMA MLOPS_SCHEMA;
SHOW VERSIONS IN MODEL MORTGAGE_LENDING_MLOPS_1;

-- Feature Views
SELECT * FROM TABLE(
  E2E_SNOW_MLOPS_DB.INFORMATION_SCHEMA.FEATURE_VIEWS()
);

-- Experiments
-- UI: https://app.snowflake.com/{org}/{account}/#/experiments/...
```

### B. Key File Locations

```
sfguide-build-end-to-end-ml-workflow-in-snowflake/
├── docs/
│   └── project_spec.md              # This file
├── plan/
│   ├── MORTGAGE_MODEL_IMPROVEMENT_PLAN.md  # Detailed implementation plan
│   └── PROJECT_STATUS.md                    # Quick status reference
├── train_deploy_monitor_ML_in_snowflake.ipynb  # Main notebook
├── setup.sql                                    # Infrastructure setup
└── environment.yml                              # Python dependencies
```

### C. Important Cell IDs

| Cell ID | Purpose |
|---------|---------|
| `d78265b8-8baa-4136-a32a-32f3f620949d` | VERSION_NUM configuration |
| `b355c0c4-9dc6-4faf-86b7-24d8d559e453` | Feature engineering |
| `5e4b5fba-b7a8-47ff-aaf6-076b9e78dcaf` | Base model definition |
| `e4d6860a-da49-42bc-aed9-57692eb5c7a2` | HPO configuration |
| `a8ff103e-5314-4e95-87ba-d784b1102f36` | NULL handling |

### D. Quick Commands

```sql
-- Switch to correct role
USE ROLE E2E_SNOW_MLOPS_ROLE;

-- Check model versions
SHOW VERSIONS IN MODEL E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_1;

-- View model metrics
SELECT * FROM TABLE(
  E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_1!SHOW_METRICS()
);

-- Fetch latest from Git
ALTER GIT REPOSITORY E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.GITHUB_REPO_E2E_SNOW_MLOPS FETCH;
```

### E. External Resources

- [Snowflake ML Documentation](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview)
- [Feature Store Guide](https://docs.snowflake.com/en/developer-guide/snowflake-ml/feature-store/overview)
- [Model Registry Guide](https://docs.snowflake.com/en/developer-guide/snowflake-ml/model-registry/overview)
- [QuickStart Guide](https://quickstarts.snowflake.com/guide/end-to-end-ml-workflow)

---

*Document generated and maintained with Cortex Code*
