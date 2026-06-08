# PHASE 1: Project Design & Model Development

## Overview
Phase 1 establishes the foundation for your MLOps project. This phase covers project planning, initial code organization, team collaboration setup, data handling, baseline model development, and comprehensive documentation. By the end of this phase, you should have a well-organized repository with a trained baseline model and clear documentation for future team members.

---

## 1. Project Proposal

- [x] **Scope & Objectives**: Define the problem statement, goals, and success metrics for TeamArtemisSE489
- [x] **Detailed Description**: Write a 300+ word project description covering the business context, technical approach, and expected outcomes
- [x] **Dataset Selection**: Choose appropriate dataset(s) and document the selection justification
- [x] **Dataset Description**: Document dataset characteristics (size, features, format, sources)
- [x] **Model Considerations**: Identify initial model architectures and algorithms suitable for your problem
- [x] **Open-Source Tools**: Document and justify the selection of open-source tools and libraries for the project

This is available in the [requirements](README.txt) file at the top.

---

## 2. Code Organization & Setup

- [x] **GitHub Repository**: Create repository with cookiecutter MLOps structure
  - A cookie-cutter template was used to make this very repo. Only a few changes were needed.
- [x] **Environment Setup**: Configure Python virtual environment (venv or conda)
  -done
- [x] **Dependency Management**: Create and maintain requirements.txt or pyproject.toml
  - the [requirements](requirements.txt) file and [pyproject](pyproject.toml).
- [x] **Project Structure**: Organize code with clear separation of concerns (src/, tests/, data/, etc.)
  - done
- [x] **Version Pinning**: Pin all critical dependencies to specific versions
  - done
- [x] **Installation Documentation**: Document how to set up the development environment
  - done

---

## 3. Version Control & Collaboration

- [x] **Regular Commits**: Establish commit discipline with descriptive, atomic commits
  - Pull requests are used for submission. Angel Hernandez is responsible for reviewing each request and resolving conflicts.
- [x] **Branching Strategy**: Implement feature branching (e.g., git-flow or GitHub Flow)
  - simple usage on a per-change basis, frequently to avoid conflicts.
- [x] **Pull Request Process**: Establish PR template and review requirements
  - Simple but effective usage of pull requests and using Discord for communication.
- [x] **Team Roles**: Clearly define responsibilities (author: TeamArtemisIV, team members, reviewers)
  - available in the [README](src/README.txt) file.
- [x] **Code Review Guidelines**: Document code review expectations and checklist
  - All submissions are accompanied by communication on Discord. Due to our small team, it is faster to communicate through Discord rather than large commit messages.
- [x] **Commit History**: Maintain clean, readable git history for project traceability
  - available on this repo's history.

---

## 4. Data Handling

- [x] **Data Cleaning Scripts**: Create reproducible scripts for data cleaning and preprocessing
- [x] **Normalization**: Implement feature normalization/standardization with proper documentation
- [x] **Data Augmentation**: Develop and document data augmentation strategies if applicable
- [x] **Data Documentation**: Create a data dictionary with feature descriptions and data types
- [x] **Data Splits**: Define and implement train/validation/test split strategy
- [x] **Data Validation**: Create scripts to validate data quality and consistency
- [x] **DVC Setup (Optional)**: Initialize DVC for data versioning if managing large datasets

---

## 5. Model Training

- [x] **Training Environment**: Set up local/cloud training environment with GPU support if needed
- [x] **Baseline Model**: Implement and train a baseline model
- [x] **Hyperparameter Configuration**: Document baseline hyperparameters and rationale
- [x] **Evaluation Metrics**: Define and calculate relevant metrics (accuracy, F1, RMSE, etc.)
- [x] **Model Persistence**: Save trained models with version information
- [x] **Training Reproducibility**: Ensure training is reproducible (seed management, logging)
- [x] **Performance Baseline**: Document baseline model performance as reference point

---

## 6. Documentation & Reporting

- [x] **README**: Create comprehensive README with:
  - [x] Project overview and objectives
  - [x] Setup and installation instructions
  - [x] Quick start guide for running training
  - [x] Dependencies and requirements
  - [x] Contributing guidelines
  - [x] License information
- [x] **Code Docstrings**: Add docstrings to all functions and classes (NumPy/Google style)
- [x] **Code Style**: Implement ruff configuration for linting
- [x] **Type Hints**: Add type hints throughout codebase
- [x] **Type Checking**: Configure mypy for static type checking
- [x] **Makefile**: Create Makefile with commands for:
  - [x] `make setup` - install dependencies
  - [x] `make train` - run training pipeline
  - [x] `make test` - run tests
  - [x] `make lint` - run linting checks
  - [x] `make format` - auto-format code
- [x] **CONTRIBUTING.md**: Document contribution guidelines and development workflow
- [x] **API Documentation**: Document all public APIs and interfaces

---

> **Checklist:** Use this as a guide for documenting your Phase 1 deliverables.
