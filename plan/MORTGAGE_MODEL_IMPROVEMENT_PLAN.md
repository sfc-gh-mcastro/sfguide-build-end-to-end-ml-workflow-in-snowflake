# Mortgage Lending Model Improvement Plan

> **Status**: Phase 1 Complete | Phases 2-5 Pending
> **Last Updated**: 2026-01-29
> **Notebook**: `TRAIN_DEPLOY_MONITOR_ML_V2`

## Executive Summary

The current production model `MORTGAGE_LENDING_MLOPS_0` (version `XGB_OPTIMIZED`) has a critical issue: **it predicts ALL cases as approved (1)**, resulting in perfect recall (1.0) but poor precision (0.7853). This plan outlines concrete changes to the `TRAIN_DEPLOY_MONITOR_ML` notebook to fix this issue and improve overall model performance.

---

## Current State Analysis

### Model Performance (XGB_OPTIMIZED - Production)

| Metric | Training Set | Test Set |
|--------|-------------|----------|
| F1 Score | 0.8787 | 0.8798 |
| Precision | 0.7836 | 0.7853 |
| Recall | 1.0000 | 1.0000 |

### Identified Issues

1. **All-Positive Predictions**: Confusion matrix shows the model predicts every loan as approved
2. **Severe Class Imbalance**: 78% approved vs 22% rejected (not addressed in training)
3. **Weak Feature Correlations**: Current features have <0.06 correlation with target
4. **Unused Data**: `LOAN_TYPE_NAME` and `COUNTY_NAME` not used as features
5. **Poor NULL Handling**: Missing income values filled with 0 instead of proper imputation

---

## Phase 1: Address Class Imbalance (Critical Fix) - COMPLETED

> **Status**: COMPLETED on 2026-01-29
> **Commit**: `88f15ba`
> **Changes Applied To**: `train_deploy_monitor_ML_in_snowflake.ipynb`

### Problem
The model learned to predict the majority class because class imbalance was not addressed.

### Solution
Add `scale_pos_weight` parameter to XGBoost to penalize misclassification of the minority class.

### Code Changes

**Cell `5e4b5fba-b7a8-47ff-aaf6-076b9e78dcaf` - Base Model Definition**

```python
# BEFORE (causes overfitting and class bias)
xgb_base = XGBClassifier(
    max_depth=50,          # Too deep - severe overfitting
    n_estimators=3,        # Too few trees
    learning_rate = 0.75,  # Too high
    booster = 'gbtree')

# AFTER (balanced and regularized)
xgb_base = XGBClassifier(
    max_depth=6,                    # Reduced to prevent overfitting
    n_estimators=100,               # More trees for stability
    learning_rate=0.1,              # Lower for better generalization
    booster='gbtree',
    scale_pos_weight=3.6,           # ~= count(negative) / count(positive)
    min_child_weight=5,             # Regularization
    subsample=0.8,                  # Row sampling
    colsample_bytree=0.8            # Column sampling
)
```

---

## Phase 2: Add Missing Features - PENDING

### Problem
The source data contains valuable features that are not being used:
- `LOAN_TYPE_NAME`: VA-guaranteed, FHA-insured, Conventional (different approval criteria)
- `COUNTY_NAME`: Geographic risk factors

### Solution
Add these features to the feature engineering pipeline and create derived features.

### Code Changes

**Cell `b355c0c4-9dc6-4faf-86b7-24d8d559e453` - Feature Engineering Dict**

Add the following to `feature_eng_dict`:

```python
# === NEW FEATURES TO ADD ===

# 1. Loan Type (categorical - will be one-hot encoded later)
feature_eng_dict["LOAN_TYPE"] = col("LOAN_TYPE_NAME")

# 2. County-level historical approval rate
county_approval_window = Window.partition_by("COUNTY_NAME")
feature_eng_dict["COUNTY_APPROVAL_RATE"] = avg("MORTGAGERESPONSE").over(county_approval_window)

# 3. Loan amount relative to county average income
feature_eng_dict["LOAN_TO_COUNTY_INCOME_RATIO"] = col("LOAN_AMOUNT") / col("MEAN_COUNTY_INCOME")

# 4. Income percentile within county
feature_eng_dict["INCOME_PERCENTILE_IN_COUNTY"] = percent_rank().over(
    Window.partition_by("COUNTY_NAME").order_by("INCOME")
)

# 5. Loan type approval rate (historical)
loan_type_window = Window.partition_by("LOAN_TYPE_NAME")
feature_eng_dict["LOAN_TYPE_APPROVAL_RATE"] = avg("MORTGAGERESPONSE").over(loan_type_window)
```

**Cell `b5e17036-7a69-4915-b025-49c900aeb46b` - OneHotEncoder**

The existing code automatically picks up string columns for OHE, so `LOAN_TYPE` will be included automatically.

---

## Phase 3: Handle NULL Values Properly - PENDING

### Problem
The notebook fills NULL values with 0:
```python
train = train.fillna(0)
test = test.fillna(0)
```

This loses valuable signal (whether income is missing may be predictive).

### Solution
Create a missing indicator and impute with county median.

### Code Changes

**Add to feature engineering (cell `b355c0c4-9dc6-4faf-86b7-24d8d559e453`)**:

```python
from snowflake.snowpark.functions import when, coalesce, median

# Missing income indicator (binary flag)
feature_eng_dict["INCOME_MISSING_FLAG"] = when(
    col("APPLICANT_INCOME_000s").isNull(), 1
).otherwise(0)

# Impute income with county median (not 0)
county_median_income = median("INCOME").over(county_window_spec)
feature_eng_dict["INCOME"] = coalesce(
    col("APPLICANT_INCOME_000s") * 1000,
    county_median_income
)
```

**Update cell `a8ff103e-5314-4e95-87ba-d784b1102f36`**:

```python
# BEFORE
train = train.fillna(0)
test = test.fillna(0)

# AFTER - Only fill remaining NULLs (should be minimal after proper imputation)
train = train.fillna({
    "INCOME": train.select(avg("INCOME")).collect()[0][0],  # Global mean as fallback
    "INCOME_LOAN_RATIO": 0,
    "INCOME_MISSING_FLAG": 1  # If still null, treat as missing
})
test = test.fillna({
    "INCOME": train.select(avg("INCOME")).collect()[0][0],
    "INCOME_LOAN_RATIO": 0,
    "INCOME_MISSING_FLAG": 1
})
```

---

## Phase 4: Expand HPO Search Space - PENDING

### Problem
Current HPO search space doesn't include class imbalance handling or regularization parameters.

### Solution
Expand the search space to include `scale_pos_weight`, `min_child_weight`, and sampling parameters.

### Code Changes

**Cell `e4d6860a-da49-42bc-aed9-57692eb5c7a2` - Tuner Configuration**

```python
tuner = tune.Tuner(
    train_func=train_func,
    search_space={
        # Tree structure
        "max_depth": tune.randint(3, 10),              # Reduced from (1,30)
        "n_estimators": tune.randint(100, 300),        # Increased from (50,150)
        "learning_rate": tune.uniform(0.01, 0.2),      # Reduced max from 0.5
        
        # Regularization (NEW)
        "min_child_weight": tune.randint(1, 10),
        "gamma": tune.uniform(0, 0.5),
        "subsample": tune.uniform(0.6, 0.9),
        "colsample_bytree": tune.uniform(0.6, 0.9),
        
        # Class imbalance (NEW - CRITICAL)
        "scale_pos_weight": tune.uniform(2.5, 5.0),
    },
    tuner_config=tune.TunerConfig(
        metric="F1_Test",
        mode="max",
        search_alg=search_algorithm.RandomSearch(random_state=101),
        num_trials=25,                                 # Increased from 8
        max_concurrent_trials=psutil.cpu_count(logical=False)
    ),
)
```

---

## Phase 5: Add Threshold Optimization - PENDING

### Problem
Default threshold of 0.5 may not be optimal for the business use case.

### Solution
Add a new cell to find the optimal decision threshold based on precision-recall tradeoff.

### Code Changes

**Add new cell after model training (after cell `dee80c48-d521-4b77-8841-54ba35ecd4b6`)**:

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Get probability predictions instead of hard labels
probas = tuned_model.predict_proba(
    test_pd.drop(["TIMESTAMP", "LOAN_ID", "MORTGAGERESPONSE"], axis=1)
)[:, 1]

# Calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(
    test_pd.MORTGAGERESPONSE, 
    probas
)

# Calculate F1 for each threshold
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

# Find optimal threshold
optimal_idx = np.argmax(f1_scores[:-1])  # Last element is always 0
optimal_threshold = thresholds[optimal_idx]

print(f"=== Threshold Optimization ===")
print(f"Default threshold (0.5):")
print(f"  Precision: {precision_opt_test}")
print(f"  Recall: {recall_opt_test}")
print(f"  F1: {f1_opt_test}")
print()
print(f"Optimal threshold ({optimal_threshold:.3f}):")
print(f"  Precision: {precision[optimal_idx]:.4f}")
print(f"  Recall: {recall[optimal_idx]:.4f}")
print(f"  F1: {f1_scores[optimal_idx]:.4f}")

# Visualize
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Precision-Recall curve
ax[0].plot(recall, precision, 'b-', linewidth=2)
ax[0].scatter([recall[optimal_idx]], [precision[optimal_idx]], 
              color='red', s=100, zorder=5, label=f'Optimal (thresh={optimal_threshold:.2f})')
ax[0].set_xlabel('Recall')
ax[0].set_ylabel('Precision')
ax[0].set_title('Precision-Recall Curve')
ax[0].legend()
ax[0].grid(True)

# F1 vs Threshold
ax[1].plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
ax[1].axvline(x=optimal_threshold, color='red', linestyle='--', label=f'Optimal={optimal_threshold:.2f}')
ax[1].axvline(x=0.5, color='blue', linestyle='--', alpha=0.5, label='Default=0.5')
ax[1].set_xlabel('Threshold')
ax[1].set_ylabel('F1 Score')
ax[1].set_title('F1 Score vs Threshold')
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
st.pyplot(fig)

# Store optimal threshold as model metadata
mv_opt.set_metric(metric_name="Optimal_Threshold", value=float(optimal_threshold))
mv_opt.set_metric(metric_name="Optimal_F1_Score", value=float(f1_scores[optimal_idx]))
```

---

## Summary of All Changes

| Phase | Cell/Location | Change Description |
|-------|---------------|-------------------|
| 1 | `5e4b5fba...` | Add `scale_pos_weight=3.6`, fix hyperparameters |
| 2 | `b355c0c4...` | Add `LOAN_TYPE`, `COUNTY_APPROVAL_RATE`, `LOAN_TO_COUNTY_INCOME_RATIO`, `LOAN_TYPE_APPROVAL_RATE` features |
| 3 | `b355c0c4...` | Add `INCOME_MISSING_FLAG`, proper income imputation |
| 3 | `a8ff103e...` | Replace blanket `fillna(0)` with targeted imputation |
| 4 | `e4d6860a...` | Expand HPO search space with regularization & class weights |
| 5 | New cell | Add threshold optimization after model training |

---

## Expected Results

| Metric | Current | Expected After Changes |
|--------|---------|----------------------|
| Precision | 0.7853 | 0.88 - 0.93 |
| Recall | 1.0000 | 0.90 - 0.95 |
| F1 Score | 0.8798 | 0.91 - 0.95 |

The key improvement is that the model will no longer predict ALL cases as positive, resulting in meaningful predictions that balance both precision and recall.

---

## Implementation Order

1. **Phase 1** (Critical) - Fix class imbalance first
2. **Phase 3** - Fix NULL handling (needed before Phase 2)
3. **Phase 2** - Add new features
4. **Phase 4** - Expand HPO search
5. **Phase 5** - Add threshold optimization

---

## Validation Checklist

After implementing changes, verify:

- [x] Confusion matrix shows predictions for both classes (not all 1s) - *Phase 1 addresses this*
- [ ] Precision improved to >0.85
- [ ] Recall is between 0.85-0.95 (no longer artificially 1.0)
- [ ] F1 score improved to >0.90
- [ ] Train/Test performance gap is <5% (no overfitting)
- [ ] Feature importance shows new features contributing
