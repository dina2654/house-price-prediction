# House Sale Price Prediction Engine & Analysis

An end-to-end Machine Learning pipeline and exploratory data analysis (EDA) for predicting residential property prices. This repository combines visual data storytelling in a Jupyter Notebook with a modular software engineering structure (`src/`) for scalable preprocessing and model training.

---

## 📌 Executive Summary

Predicting residential house prices requires handling mixed data types, missing entries, non-linear relationships, and high feature collinearity. 

This project evaluates baseline **Multivariable Linear Regression** against a hyperparameter-tuned **Ridge Regression (`RidgeCV`)** model. By applying $L_2$ regularization, the model mitigates variance caused by multicollinearity—validated through **Variance Inflation Factor (VIF)** analysis—to deliver robust valuation predictions on held-out test data.

---

## 📁 Repository Structure

```text
house-price-prediction/
├── data/
│   └── HousePricesDataset.xlsx       # Housing dataset
├── notebooks/
│   └── house_price_analysis.ipynb    # Visual EDA & diagnostic research lab
├── src/
│   ├── data_preprocessing.py         # Modular data loading, imputation, scaling & VIF
│   └── train_model.py                # Pipeline construction, RidgeCV tuning & evaluation
├── .gitignore                        # Git exclusion rules
├── requirements.txt                  # Environment dependencies
└── README.md                         # Executive project documentation
```
## Key Technical Highlights

Data Cleaning & Pipeline Engineering: Constructed custom ColumnTransformer pipelines inside scikit-learn incorporating SimpleImputer (median for numeric, most frequent for categorical), StandardScaler, and OneHotEncoder(handle_unknown='ignore').

Multicollinearity Diagnostics: Analyzed Variance Inflation Factors (VIF) on transformed numeric and one-hot encoded features to isolate redundant price drivers.

Regularization & Hyperparameter Tuning: Automated 5-fold cross-validated grid search over 50 regularization penalties ($\alpha \in [10^{-2}, 10^3]$) via RidgeCV.

Software Engineering Best Practices: Separated data transformation and training workflows into clean, reusable Python modules with execution blocks (if __name__ == "__main__":).

## Model Evaluation & Results

Models were trained on an 80/20 train-test split ($N_{\text{train}} = 1,168$, $N_{\text{test}} = 292$).

| Model | RMSE ($) | MAE ($) | R2 Score |
|---|---:|---:|---:|
| Linear Regression (Baseline) | $34,714.70 | $20,156.87 | 0.7712 |
| Ridge Regression (Optimal α) | $34,466.05 | $19,300.42 | 0.7745 |



## 🔑 Primary Valuation Drivers


### Top 5 Positive Price Drivers (Increases Value)
1. **Northridge Heights Location (`Neighborhood_NridgHt`)**: +$32,743.88
2. **Stone Brook Location (`Neighborhood_StoneBr`)**: +$29,976.17
3. **Northridge Location (`Neighborhood_NoRidge`)**: +$25,960.02
4. **Overall Material & Finish Quality (`OverallQual`)**: +$19,849.88 per unit increase
5. **Above Grade Living Area (`GrLivArea`)**: +$16,621.10 per standard deviation increase

### Top 5 Negative Price Drivers (Decreases Value)
1. **Townhouse Building Type (`BldgType_Twnhs`)**: -$21,205.61
2. **Northwest Ames Location (`Neighborhood_NWAmes`)**: -$20,797.31
3. **Highly Irregular Lot Shape (`LotShape_IR3`)**: -$20,552.62
4. **Gilbert Location (`Neighborhood_Gilbert`)**: -$19,714.18
5. **Bloomington Heights Location (`Neighborhood_Blmngtn`)**: -$13,322.69





