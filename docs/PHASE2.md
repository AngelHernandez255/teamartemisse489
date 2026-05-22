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
We integrated Weights & Biases (W&B) into our training workflow to track hyperparameters, evaluation metrics, dataset metadata, and the trained model artifact making it easy to compare runs, reproduce the final configuration, and understand how the best model was selected.

W&B was used in three ways:
- Run tracking: each training run logged its configuration, metrics, and summary values
- Sweep management: the grid sweep explored multiple SVD configurations in a structured way
- Artifact tracking: the trained model was saved and logged as a W&B artifact so the best version can be traced later

In each run, we tracked:
- Hyperparameters: `n_factors`, `n_epochs`, `lr_all`, `reg_all`
- Training settings: `random_state`, `test_size`, `k`, and the relevance threshold used for top-k evaluation
- Dataset metadata: `num_users`, `num_movies`, `num_ratings`, `rating_column`, `rating_scale_min`, `rating_scale_max`, and whether target-rating preprocessing was used
- Evaluation metrics: `precision_at_10`, `recall_at_10`, `rmse`, `mae`, and `training_time`
- Diagnostic distributions: `precision_distribution`, `recall_distribution`, and the top-k counts `tp`, `fp`, `fn`
- Artifacts: the trained model file `models/svd.joblib` and the metrics snapshot `models/svd_metrics.json`

The sweep was configured as a grid search over the SVD hyperparameters:
- `n_factors`: latent factor size
- `n_epochs`: number of training epochs
- `lr_all`: learning rate
- `reg_all`: regularization strength

The sweep objective was to maximize `precision_at_10`, since our main goal was to improve the quality of the top-10 recommendations. We used `rmse` and `training_time` as secondary comparison signals when choosing the final run.

Setup used:
- `wandb login`
- `python -m teamartemisse489.train_model`
- `wandb sweep sweep.yaml`
- `wandb agent sakshigorkhaliprojects/Team-Artemisse489-Recommender/3zogxl0j`

Team project:
- https://wandb.ai/sakshigorkhaliprojects/Team-Artemisse489-Recommender

Shared report:
- https://api.wandb.ai/links/sakshigorkhaliprojects/f0tplq6e

![Runs table from W&B](../src/teamartemisse489/visualization/wandb_runs_sorted_precision_at_10.png)

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
We used the baseline notebook and the W&B sweep results together to evaluate model quality and choose the final SVD configuration. The first comparison happened in [notebooks/02_Baseline_Models.ipynb](notebooks/02_Baseline_Models.ipynb), where we tested several candidate recommenders and compared them on both prediction and ranking behavior. SVD gave the strongest initial balance of performance, stability, and reproducibility, so it became the model we tuned further with the W&B sweep.

The main selection criterion was `precision_at_10`, because our goal is to rank the most relevant items near the top of the recommendation list. We used `rmse` and `training_time` as secondary criteria to make sure the final model was still accurate on rating prediction and efficient enough to train repeatedly during sweep experiments.

- Model comparison results: SVD outperformed the other baseline candidates for the top-k ranking objective
- Hyperparameter sensitivity: the sweep showed that larger latent dimensionality and more epochs generally improved ranking quality within the tested search space
- Best-run evidence: the sweep table screenshot below is the reference used to verify `n_factors=100`, `n_epochs=30`, `lr_all=0.01`, `reg_all=0.1`
- Comparison columns used in W&B: `precision_at_10`, `rmse`, `mae`, and `training_time`

How the best hyperparameters were selected:
- We opened the W&B runs table and sorted by `precision_at_10` in descending order
- We compared the top runs against `rmse`

Best model results: SVC
- `n_factors=100`
- `n_epochs=30`
- `lr_all=0.01`
- `reg_all=0.1`
- `precision_at_10=0.7687069`
- `recall_at_10=0.7790209`
- `rmse=1.0819889`
- `mae=0.863545`
- `training_time=5.15s`

Shared W&B report:
- https://api.wandb.ai/links/sakshigorkhaliprojects/f0tplq6e

### 4. Model Artifacts
- Best model saved and versioned
- Model evaluation report
- Training curves and visualizations
- Configuration documentation


## Hydra Configuration Management

Integrated Hydra for centralized and modular configuration management across the MLOps pipeline.

### Successful Hydra Training Run

This demonstrates successful Hydra configuration loading, model training, and experiment tracking integration.

![Hydra Training Output](docs/screenshots/mlops_hydra.png)
![Hydra Training Output](docs/screenshots/hydra2.png)
![Hydra Training Output](docs/screenshots/hydra3.png)
---

### Configuration Validation Example

The system validates runtime configuration values and prevents invalid experiment execution.

![Hydra Validation Output](docs/screenshots/config_validation.png)

---

### Configuration Version Control

All Hydra configuration files were version-controlled alongside the application source code using Git and GitHub collaborative workflows.

Implemented:
- feature branch workflow
- pull requests
- merge conflict resolution
- synchronized configuration updates with training pipeline


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
