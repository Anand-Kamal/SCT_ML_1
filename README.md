# House Price Prediction — Linear Regression

Predicting residential sale prices using linear regression on the Kaggle
["House Prices - Advanced Regression Techniques"](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
dataset (Ames, Iowa housing data).

## Overview

This project builds a linear regression model to predict `SalePrice` from a
set of numeric features selected by their correlation with the target,
rather than guessing which columns matter up front. Several improvements
were tested with 5-fold cross-validation before settling on a final model:

| Approach                                          | CV R²  |
|----------------------------------------------------|--------|
| Baseline (7 correlated features, raw target)        | 0.739  |
| + remove known dataset outliers                     | 0.809  |
| + log-transform the target                          | 0.878  |
| **+ additional features + Ridge regularization**    | **0.886** |

**Final result:** Test R² ≈ 0.88, RMSE ≈ $25,800 on a held-out 20% split.

## Methodology

1. **Feature selection** — start from all numeric columns with `|correlation| > 0.5`
   against `SalePrice` (`OverallQual`, `GrLivArea`, `GarageCars`, `GarageArea`,
   `TotalBsmtSF`, `1stFlrSF`, `FullBath`, `TotRmsAbvGrd`, `YearBuilt`,
   `YearRemodAdd`).
2. **Feature engineering** — add `HouseAge` (`YrSold - YearBuilt`) and
   `TotalSF` (total finished square footage across basement + both floors),
   plus `Fireplaces` and `MasVnrArea`.
3. **Outlier removal** — drop two documented Ames dataset outliers (`Id` 524
   and 1299): unusually large homes sold far below market value in what are
   understood to be non-standard sales.
4. **Log-transform the target** — `SalePrice` is right-skewed, so the model
   is trained on `log1p(SalePrice)` and predictions are converted back with
   `expm1`.
5. **Ridge regression** — several features are correlated with each other
   (e.g. `GarageCars`/`GarageArea`), which destabilizes plain OLS
   coefficients. `RidgeCV` picks a regularization strength via
   cross-validation to handle this.
6. **Evaluation** — MAE, RMSE, and R² on train/test splits, plus 5-fold
   cross-validation for a more robust estimate than a single split.

## Repository Structure

```
.
├── house_price_corr_best.ipynb   # main notebook (Google Colab-ready)
├── train.csv                     # Kaggle training data (not included — see below)
├── test.csv                      # Kaggle test data (not included — see below)
├── submission.csv                # generated predictions for Kaggle submission
└── README.md
```

## Getting the Data

The dataset isn't included in this repo. Download `train.csv` and `test.csv`
from the [Kaggle competition page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
(free Kaggle account required) and place them alongside the notebook, or
upload them when prompted if running in Colab.

## Running the Notebook

**Option 1 — Google Colab (recommended)**
1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `house_price_corr_best.ipynb`
3. Run all cells — the second cell will prompt you to upload `train.csv` and
   `test.csv`

**Option 2 — Locally / Jupyter**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook house_price_corr_best.ipynb
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Results

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| MAE    | $18,226      | $18,920  |
| RMSE   | $26,468      | $25,764  |
| R²     | 0.892        | 0.880    |

Train and test scores are close together, which suggests the model
generalizes reasonably well rather than overfitting the training data.

## Notes & Limitations

- This model is intentionally restricted to a small set of numeric features
  and stays within a linear regression framework. Tree-based or ensemble
  models (e.g. gradient boosting) typically score noticeably higher on this
  dataset, but that's outside the scope of this project.
- Because features are standardized and the target is log-transformed,
  model coefficients represent relative (percentage-scale) effects rather
  than direct dollar amounts per unit.

## Acknowledgments

Dataset: Dean De Cock, "Ames Housing dataset," used in the Kaggle
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
competition.
