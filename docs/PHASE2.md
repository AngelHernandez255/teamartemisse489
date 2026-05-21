# Phase 2: Model Development

## Overview
This phase focuses on building, training, and validating machine learning models.

## Objectives

- [ ] Implement baseline model
- [ ] Train and evaluate initial models
- [ ] Hyperparameter tuning
- [ ] Cross-validation and performance analysis
- [ ] Model comparison and selection

## Deliverables

### 1. Model Implementation
- Model architecture defined
- Training pipeline implemented
- Evaluation metrics chosen
- Baseline performance established

### 2. Experiment Tracking
- All experiments logged and documented
- MLflow experiment tracking configured

### 2. Debugging Practices
- Debugger tools: `pdb`, VS Code Python Debugger, and the trainer's `--debug`
  pause point
- Implementation: `models/train.py`
- Validation coverage: missing parquet file, empty dataframe, missing required
  columns, null required values, nonnumeric ratings, and ratings outside the
  expected 1-5 range

Run line-by-line debugging:
```bash
python -m pdb models/train.py
```

Run until the dataframe is loaded and validated, then pause:
```bash
python models/train.py --debug
```

Useful debugger expressions:
```python
p df.shape
p df.dtypes
p df[["userId", "movieId", "rating"]].head()
p df["rating"].describe()
```

Scenario 1: if `userId`, `movieId`, or `rating` is missing, training raises a
clear validation error before Surprise receives the dataframe. Inspect
`df.columns` and fix the data preparation step.

Scenario 2: if ratings are null, nonnumeric, or outside 1-5, training raises a
clear validation error before fitting. Inspect `df["rating"].describe()` and
the invalid rows to correct preprocessing.

### 3. Performance Analysis
- Model comparison results
- Hyperparameter sensitivity analysis
- Feature importance analysis
- Error analysis and patterns

### 4. Model Artifacts
- Best model saved and versioned
- Model evaluation report
- Training curves and visualizations
- Configuration documentation

## Model Selection

*To be filled in during Phase 2*

### Chosen Model
- Model Type: 
- Best Hyperparameters: 
- Performance Metrics: 

## Key Results

*To be filled in during Phase 2*

## Challenges and Solutions

*To be filled in during Phase 2*

## Next Steps

Move to Phase 3 once model is selected and meets performance requirements.

## Status

- Start Date: 
- Estimated Completion: 
- Actual Completion: 
- Status: Not Started
