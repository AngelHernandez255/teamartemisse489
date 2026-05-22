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
We built the recommendation pipeline around a Surprise-based SVD model. The baseline comparison was first explored in [notebooks/02_Baseline_Models.ipynb](notebooks/02_Baseline_Models.ipynb), where we evaluated several candidate recommendation methods and compared them on both ranking and prediction quality. SVD gave the strongest initial balance of performance and reproducibility, so it became the model we carried forward into optimizing, hyperparameter tuning and final evaluation. The training pipeline itself is implemented in [src/teamartemisse489/train_model.py](src/teamartemisse489/train_model.py), which handles data preparation, train/test splitting, top-k evaluation, metric logging, and artifact saving so the model can be retrained and compared consistently.

- Model architecture: collaborative filtering with SVD latent factors
- Training pipeline: end-to-end script in [src/teamartemisse489/train_model.py](src/teamartemisse489/train_model.py)
- Evaluation metrics:
  - `precision_at_10`: measures how many of the top 10 recommended items are actually relevant; this is our main ranking metric
  - `recall_at_10`: measures how many relevant items were recovered in the top 10; useful for checking whether the model is missing too many good recommendations
  - `rmse`: measures average prediction error on rating values; this helps judge how well the model estimates explicit ratings
  - `mae`: similar to RMSE but less sensitive to large errors; useful as a second prediction-quality check
  - `training_time`: measures how long the model takes to train; important for reproducibility and sweep efficiency
- Baseline performance: the baseline notebook showed that SVD had the best initial performance among the candidate models, so we used it as the main model for hyperparameter tuning through the W&B sweep

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
