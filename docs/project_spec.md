# Mortgage Lending MLOps - Project Specification

> **Version**: 1.0  
> **Last Updated**: 2026-01-30

---

## Table of Contents

1. [Overview](#overview)
2. [Business Requirements](#business-requirements)
3. [Data Specification](#data-specification)
4. [Feature Engineering](#feature-engineering)
5. [Model Specification](#model-specification)
6. [API Reference](#api-reference)
7. [Implementation Guide](#implementation-guide)

---

## Overview

### Project Goal

Build and operationalize an end-to-end ML workflow in Snowflake for predicting mortgage loan approval likelihood, demonstrating Snowflake's ML platform capabilities including Feature Store, Model Registry, Distributed HPO, Experiment Tracking, and Model Monitoring.

### Use Case

Binary classification to predict whether a mortgage loan application will be **approved (1)** or **rejected (0)** based on applicant and loan characteristics.

### Success Criteria

| Metric | Target |
|--------|--------|
| Test F1 Score | > 0.91 |
| Test Precision | > 0.88 |
| Test Recall | 0.90 - 0.95 |
| Train/Test Gap | < 5% |

---

## Business Requirements

### Problem Statement

Financial institutions need to predict mortgage loan approval likelihood to:
- Streamline underwriting processes
- Reduce manual review overhead
- Ensure consistent decision-making
- Comply with fair lending regulations

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

### Operational Requirements

| Requirement | Target |
|-------------|--------|
| Inference Latency | < 100ms |
| Model Drift Alert | Weekly |
| Feature Freshness | Daily |
| Explainability | All predictions |

---

## Data Specification

### Source Data

**Table**: `MORTGAGE_LENDING_DEMO_DATA`

| Column | Type | Description | Nullable |
|--------|------|-------------|----------|
| LOAN_ID | INTEGER | Unique loan identifier | No |
| TS | TIMESTAMP | Loan application timestamp | No |
| LOAN_AMOUNT_000s | FLOAT | Loan amount in thousands | No |
| APPLICANT_INCOME_000s | FLOAT | Applicant income in thousands | Yes (~15%) |
| LOAN_PURPOSE_NAME | STRING | Purpose: Purchase, Refinancing, Home improvement | No |
| LOAN_TYPE_NAME | STRING | Type: Conventional, FHA-insured, VA-guaranteed | No |
| COUNTY_NAME | STRING | County of property | No |
| MORTGAGERESPONSE | INTEGER | Target: 1=Approved, 0=Rejected | No |

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
| Class imbalance | Model bias toward majority class | Phase 1: scale_pos_weight (DONE) |
| Weak feature correlations | Low predictive power | Phase 2: Add derived features |

---

## Feature Engineering

### Current Features (v1)

| Feature | Type | Description | Source |
|---------|------|-------------|--------|
| MONTH | INT | Month of loan application (1-12) | Derived from TS |
| DAY_OF_YEAR | INT | Day of year (1-365) | Derived from TS |
| DOTW | INT | Day of week (0-6) | Derived from TS |
| LOAN_AMOUNT | FLOAT | Loan amount in USD | LOAN_AMOUNT_000s × 1000 |
| INCOME | FLOAT | Applicant income in USD | APPLICANT_INCOME_000s × 1000 |
| INCOME_LOAN_RATIO | FLOAT | Income / Loan Amount | Derived |
| MEAN_COUNTY_INCOME | FLOAT | Avg income by county | Window function |
| HIGH_INCOME_FLAG | INT | 1 if Income > County avg | Derived |
| AVG_THIRTY_DAY_LOAN_AMOUNT | FLOAT | 30-day rolling avg loan | Window function |
| LOAN_PURPOSE_NAME_* | INT | One-hot encoded purpose | OneHotEncoder |

### Planned Features (Phase 2)

| Feature | Type | Description | Expected Impact |
|---------|------|-------------|-----------------|
| LOAN_TYPE | STRING | VA/FHA/Conventional | High |
| COUNTY_APPROVAL_RATE | FLOAT | Historical approval rate by county | Medium |
| LOAN_TO_COUNTY_INCOME_RATIO | FLOAT | Loan / County avg income | Medium |
| INCOME_PERCENTILE_IN_COUNTY | FLOAT | Income rank within county | Medium |
| LOAN_TYPE_APPROVAL_RATE | FLOAT | Approval rate by loan type | Medium |

### Planned Features (Phase 3)

| Feature | Type | Description | Expected Impact |
|---------|------|-------------|-----------------|
| INCOME_MISSING_FLAG | INT | 1 if income was NULL | High |

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

### Architecture

| Component | Specification |
|-----------|---------------|
| Algorithm | XGBoost Classifier |
| Task | Binary Classification |
| Target | MORTGAGERESPONSE (0/1) |
| Framework | xgboost via snowflake-ml-python |

### Base Model Configuration (v1)

```python
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

### Hyperparameter Optimization

**Current Search Space:**
```python
search_space = {
    "max_depth": tune.randint(1, 30),
    "learning_rate": tune.uniform(0.01, 0.5),
    "n_estimators": tune.randint(50, 150),
}
```

**Planned Expanded Search Space (Phase 4):**
```python
search_space_expanded = {
    "max_depth": tune.randint(3, 10),
    "n_estimators": tune.randint(100, 300),
    "learning_rate": tune.uniform(0.01, 0.2),
    "min_child_weight": tune.randint(1, 10),
    "gamma": tune.uniform(0, 0.5),
    "subsample": tune.uniform(0.6, 0.9),
    "colsample_bytree": tune.uniform(0.6, 0.9),
    "scale_pos_weight": tune.uniform(2.5, 5.0),
}

tuner_config = TunerConfig(
    metric="F1_Test",
    mode="max",
    num_trials=25,
)
```

### Model Registry

```python
model_registry = Registry(
    session=session,
    database_name="E2E_SNOW_MLOPS_DB",
    schema_name="MLOPS_SCHEMA",
    options={"enable_monitoring": True}
)

mv = model_registry.log_model(
    model_name=f"MORTGAGE_LENDING_MLOPS_{VERSION_NUM}",
    model=model,
    version_name="XGB_OPTIMIZED",
    sample_input_data=train_df.limit(100),
    target_platforms=["WAREHOUSE", "SNOWPARK_CONTAINER_SERVICES"],
    options={"enable_explainability": True}
)
```

---

## API Reference

### Model Inference

**Warehouse-based inference:**
```python
# Get model version
mv = model_registry.get_model("MORTGAGE_LENDING_MLOPS_1").version("XGB_OPTIMIZED")

# Run predictions
predictions = mv.run(input_df, function_name="predict")

# Get probability scores
probabilities = mv.run(input_df, function_name="predict_proba")
```

**SQL-based inference:**
```sql
SELECT 
    LOAN_ID,
    E2E_SNOW_MLOPS_DB.MLOPS_SCHEMA.MORTGAGE_LENDING_MLOPS_1!PREDICT(...) as PREDICTION
FROM input_table;
```

### Model Explainability

```python
# Generate SHAP values
explanations = mv.run(input_df, function_name="explain")

# Visualize
from snowflake.ml.monitoring import explain_visualize
explain_visualize.plot_influence_sensitivity(explanations, features_df)
```

### Feature Store

```python
# Get feature store
fs = FeatureStore(
    session=session,
    database=DB,
    name=SCHEMA,
    default_warehouse=COMPUTE_WAREHOUSE
)

# Generate dataset with features
ds = fs.generate_dataset(
    name="MORTGAGE_DATASET",
    spine_df=df.select("LOAN_ID", "TIMESTAMP", "MORTGAGERESPONSE"),
    features=[loan_fv],
    spine_timestamp_col="TIMESTAMP",
    spine_label_cols=["MORTGAGERESPONSE"]
)
```

---

## Implementation Guide

### Phase-by-Phase Tasks

| Phase | Objective | Cell ID | Expected Impact |
|-------|-----------|---------|-----------------|
| 1 | Class imbalance | `5e4b5fba...` | DONE |
| 2 | Add features | `b355c0c4...` | +2-4% F1 |
| 3 | NULL handling | `b355c0c4...`, `a8ff103e...` | +1-2% F1 |
| 4 | Expand HPO | `e4d6860a...` | +1-3% F1 |
| 5 | Threshold optimization | New cell | +0.5-1% F1 |

### Version Control Protocol

Before implementing each phase, increment `VERSION_NUM`:

```python
# Cell: d78265b8-8baa-4136-a32a-32f3f620949d
VERSION_NUM = '2'  # Increment for next phase
```

This ensures:
- New feature views are created (not overwritten)
- New model versions are registered separately
- Clear lineage between model iterations

### Important Cell IDs

| Cell ID | Purpose |
|---------|---------|
| `d78265b8-8baa-4136-a32a-32f3f620949d` | VERSION_NUM configuration |
| `b355c0c4-9dc6-4faf-86b7-24d8d559e453` | Feature engineering |
| `5e4b5fba-b7a8-47ff-aaf6-076b9e78dcaf` | Base model definition |
| `e4d6860a-da49-42bc-aed9-57692eb5c7a2` | HPO configuration |
| `a8ff103e-5314-4e95-87ba-d784b1102f36` | NULL handling |

---

## Related Documents

- [Architecture](architecture.md) - System design and data flow
- [Project Status](project_status.md) - Current progress
- [Changelog](changelog.md) - Version history

---

*Document maintained with Cortex Code*
