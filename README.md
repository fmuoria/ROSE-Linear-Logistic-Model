# ROSE-Linear-Logistic-Model

## Overview

This repository contains loan default prediction models using traditional features (V1) and composite score-based features (V2).

## Repository Structure

```
├── notebooks/
│   ├── eda.ipynb                                    # Exploratory Data Analysis
│   ├── loan_default_models.ipynb                    # V1 Models (12 traditional features)
│   ├── loan_default_models_v2_composite_scores.ipynb # V2 Models (composite scores)
│   └── model_evaluation.ipynb                       # Comprehensive model evaluation
├── models/
│   ├── model_comparison.csv                         # Model comparison results
│   ├── v2/                                          # V2 model artifacts
│   └── [Generated files after running evaluation]
│       ├── model_evaluation_results.csv             # Complete evaluation results
│       ├── MODEL_EVALUATION_REPORT.md              # Markdown report
│       ├── roc_auc_comparison.png                  # ROC-AUC bar chart
│       ├── ks_statistic_comparison.png             # KS Statistic bar chart
│       ├── performance_heatmap.png                 # All metrics heatmap
│       ├── top5_radar_chart.png                    # Top 5 models radar chart
│       └── best_model_confusion_matrix.png         # Best model confusion matrix
└── Github Original Data.csv                         # Dataset
```

## Notebooks

### 1. EDA (eda.ipynb)
Exploratory Data Analysis with composite score definitions and visualizations.

### 2. V1 Models (loan_default_models.ipynb)
Training and evaluation of 5 models using 12 traditional features:
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

**Features (12):** Extra Income Brackets, Rent Category, School Fees Category, Age Group, Education, Loan Access, CRB Class, Income Diversity, Utility Category, Expense Ratio, Affordability HH, Living

### 3. V2 Models (loan_default_models_v2_composite_scores.ipynb)
Training and evaluation of models using composite scores across 3 feature sets:

**Feature Set A (4 features):** Composite scores only
- Financial Resilience Score
- Business Quality Score
- Stability Score
- Expense Management Score

**Feature Set B (8 features):** Composite scores + key categoricals
- All Feature Set A features
- Age Group, Education, CRB Class, Living

**Feature Set C (10 features):** Extended feature set
- All Feature Set B features
- Income Diversity (Logic on Income), Marital Status

### 4. Model Evaluation (model_evaluation.ipynb) ✨ NEW
Comprehensive evaluation notebook that:
- Trains and evaluates all 20 models (5 V1 + 15 V2)
- Calculates 6 performance metrics for each model:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
  - KS Statistic
- Generates comparison visualizations
- Exports results in multiple formats

## Running the Model Evaluation

### Prerequisites

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost imbalanced-learn matplotlib seaborn scipy
```

### Execution

1. Navigate to the notebooks directory:
   ```bash
   cd notebooks
   ```

2. Open and run `model_evaluation.ipynb` in Jupyter:
   ```bash
   jupyter notebook model_evaluation.ipynb
   ```

3. Or run all cells programmatically:
   ```bash
   jupyter nbconvert --to notebook --execute model_evaluation.ipynb --output model_evaluation_executed.ipynb
   ```

### Expected Output

After execution, the following files will be generated in the `models/` directory:
- `model_evaluation_results.csv` - Complete results table
- `MODEL_EVALUATION_REPORT.md` - Executive summary report
- 5 visualization PNG files

### Evaluation Process

The notebook:
1. Loads the dataset (`Github Original Data.csv`)
2. Generates composite scores for V2 models
3. Performs 70/15/15 train/val/test split with stratification
4. Applies SMOTE to balance training data
5. Trains all 20 models with consistent hyperparameters
6. Evaluates each model on the test set
7. Compares V1 vs V2 performance
8. Identifies the best performing model
9. Exports results and visualizations

## Model Performance

Run `model_evaluation.ipynb` to generate the latest performance metrics. Results include:

- **Best Model Identification**: Ranked by ROC-AUC score
- **V1 vs V2 Comparison**: Average metrics and improvement percentage
- **Feature Set Analysis**: Performance comparison across V2 feature sets A, B, C
- **Visualization Suite**: ROC-AUC chart, KS Statistic chart, heatmap, radar chart, confusion matrix

## Key Features

### Composite Scores (V2 Models)

1. **Financial Resilience Score (0-100)**
   - Extra Income (35%), Expense Ratio (30%), Income Diversity (20%), Savings (15%)

2. **Business Quality Score (0-100)**
   - Rent Payment (45%), Utility Expenses (30%), Business Affordability (25%)

3. **Stability Score (0-100)**
   - School Fees (40%), Regular Income (30%), Income Streams (30%)

4. **Expense Management Score (0-100)**
   - Expense Ratio (50%), Affordability HH (35%), Utility (15%)

### Evaluation Metrics

- **Accuracy**: Overall classification correctness
- **Precision**: Positive prediction accuracy (default predictions)
- **Recall**: True positive detection rate (actual defaults caught)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **KS Statistic**: Kolmogorov-Smirnov test statistic for model discrimination

## Reproducibility

All models use `random_state=42` for reproducibility. The same preprocessing steps, train/test split, and SMOTE application are applied consistently across all model training.

## License

[Add license information]

## Contributors

[Add contributor information]