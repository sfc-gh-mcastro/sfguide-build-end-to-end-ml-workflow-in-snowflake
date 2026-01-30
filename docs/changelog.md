# Mortgage Lending MLOps - Changelog

All notable changes to this project are documented in this file.

---

## [1.1.0] - 2026-01-30

### Added
- Comprehensive documentation structure in `docs/`
  - `project_spec.md` - Full requirements, API specs, tech details
  - `architecture.md` - System design and data flow diagrams
  - `project_status.md` - Current progress and next steps
  - `changelog.md` - This file

### Changed
- Reorganized documentation from `plan/` to `docs/`

---

## [1.0.0] - 2026-01-30

### Added
- **Phase 1 Implementation**: Class imbalance fix
  - Added `scale_pos_weight=3.6` to handle 78/22 class distribution
  - Added regularization parameters (`min_child_weight`, `subsample`, `colsample_bytree`)
  
- New model version: `MORTGAGE_LENDING_MLOPS_1`
  - XGB_BASE: Base model with improved configuration
  - XGB_OPTIMIZED: HPO-tuned model (production)

- Infrastructure
  - New API integration: `GITHUB_INTEGRATION_MCASTRO`
  - New Git repository pointing to fork
  - New notebook: `TRAIN_DEPLOY_MONITOR_ML_V2`

- Documentation
  - `plan/MORTGAGE_MODEL_IMPROVEMENT_PLAN.md`
  - `plan/PROJECT_STATUS.md`

### Changed
- XGBoost base model configuration:
  ```python
  # Before
  XGBClassifier(
      max_depth=50,
      n_estimators=3,
      learning_rate=0.75,
      booster='gbtree'
  )
  
  # After
  XGBClassifier(
      max_depth=6,
      n_estimators=100,
      learning_rate=0.1,
      booster='gbtree',
      scale_pos_weight=3.6,
      min_child_weight=5,
      subsample=0.8,
      colsample_bytree=0.8
  )
  ```

### Results
| Metric | Before (v0) | After (v1) | Change |
|--------|-------------|------------|--------|
| Test Precision | 0.7853 | 0.8104 | +2.51% |
| Test Recall | 1.0000 | 0.9757 | -2.43% |
| Test F1 | 0.8798 | 0.8854 | +0.56% |
| Train/Test Gap | 8.15% | 0.2% | -7.95% |

---

## [0.1.0] - 2026-01-12

### Added
- Initial model: `MORTGAGE_LENDING_MLOPS_0`
  - XGB_BASE: Baseline XGBoost classifier
  - XGB_OPTIMIZED: HPO-tuned model

- Feature Store setup
  - Entity: `LOAN_ENTITY`
  - Feature View: `Mortgage_Feature_View`

- Model Registry configuration
  - Monitoring enabled
  - Explainability enabled

- Model Monitor setup
  - Daily metric computation
  - Drift tracking

### Known Issues
- Model predicts ALL cases as approved (Recall = 1.0)
- Severe class imbalance not addressed
- Poor precision (0.7853)

---

## Planned Changes

### [1.2.0] - Phase 2: Add Missing Features
- Add `LOAN_TYPE_NAME` categorical feature
- Add `COUNTY_APPROVAL_RATE` derived feature
- Add `LOAN_TO_COUNTY_INCOME_RATIO` derived feature
- Add `INCOME_PERCENTILE_IN_COUNTY` derived feature
- Add `LOAN_TYPE_APPROVAL_RATE` derived feature

**Expected Impact**: +2-4% F1 improvement

### [1.3.0] - Phase 3: Handle NULL Values
- Add `INCOME_MISSING_FLAG` indicator
- Impute income with county median instead of 0
- Replace blanket `fillna(0)` with targeted imputation

**Expected Impact**: +1-2% F1 improvement

### [1.4.0] - Phase 4: Expand HPO Search Space
- Add `scale_pos_weight` to HPO search
- Add regularization parameters to search
- Increase trials from 8 to 25
- Narrow tree depth range (3-10)

**Expected Impact**: +1-3% F1 improvement

### [1.5.0] - Phase 5: Threshold Optimization
- Generate precision-recall curve
- Find optimal decision threshold
- Store threshold as model metadata

**Expected Impact**: +0.5-1% F1 improvement

---

## Implementation Notes

### Version Numbering

- **Model Version** (`VERSION_NUM`): Controls Snowflake object versioning
  - `'0'` → `MORTGAGE_LENDING_MLOPS_0`
  - `'1'` → `MORTGAGE_LENDING_MLOPS_1`
  - `'2'` → `MORTGAGE_LENDING_MLOPS_2` (next)

- **Changelog Version**: Follows semantic versioning
  - Major: Breaking changes or significant milestones
  - Minor: New features or phases
  - Patch: Bug fixes or minor improvements

### Commit References

| Version | Commit | Description |
|---------|--------|-------------|
| 1.1.0 | `d6c3b28` | Add comprehensive project specification |
| 1.0.0 | `fea31cd` | Add Phase 1 results with model comparison |
| 1.0.0 | `88f15ba` | Fix class imbalance in XGBoost base model |
| 0.1.0 | `feaf968` | Initial implementation |

---

## Related Documents

- [Project Spec](project_spec.md) - Requirements and API specs
- [Architecture](architecture.md) - System design and data flow
- [Project Status](project_status.md) - Current progress

---

*Document maintained with Cortex Code*
