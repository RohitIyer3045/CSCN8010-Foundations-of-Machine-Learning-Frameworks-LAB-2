# LAB 2 - MULTIVARIATE REGRESSION, NON-PARAMETRIC MODELS AND CROSS-VALIDATION

This project uses the **Scikit-Learn Diabetes Dataset** to model and predict **disease progression one year after baseline**.  
It demonstrates a full machine learning workflow — from data loading and EDA to model comparison, evaluation, and interpretation.

---

## 📁 Project Overview

**Objective:**  
Predict the *quantitative disease progression score* of diabetes patients based on 10 baseline physiological features.

**Key Steps:**
1. **Get the Data** — Load the built-in Scikit-Learn Diabetes dataset.  
2. **Frame the Problem** — Define as a supervised regression task.  
3. **EDA** — Explore distributions, correlations, and relationships.  
4. **Data Cleaning** — Validate dataset quality and normalization.  
5. **Data Splitting** — Train (75%), Validation (10%), Test (15%).  
6. **Modeling:**
   - Univariate Polynomial Regression (BMI vs target)
   - Multivariate Linear Regression (all features)
7. **Model Comparison** — Evaluate R², MAE, MAPE on train/validation/test sets.  
8. **Extended Models:** Polynomial (deg 2–3), Decision Tree, kNN, Logistic Regression.  
9. **Failure Analysis** — Identify limitations and overfitting patterns.  
10. **Conclusion** — Discuss model performance and future work.

---

## 📊 Dataset Description

**Source:** Built-in `sklearn.datasets.load_diabetes()`  

| Attribute | Description |
|------------|-------------|
| `age` | Age in years (standardized) |
| `sex` | Sex (0/1, standardized) |
| `bmi` | Body Mass Index |
| `bp` | Average Blood Pressure |
| `s1`–`s6` | Six blood serum measurements |
| **Target** | Disease progression score after one year |

**Rows:** 442 patients  
**Features:** 10 numeric (standardized)  
**Target Range:** 25 → 346  

---

## ⚙️ Model Summary

| Model | Validation R² | Validation MAE | Notes |
|:------|:--------------:|:--------------:|-------|
| Polynomial Regression (deg 1) | 0.45 | 45.5 | Best univariate |
| Multivariate Linear Regression | **0.49** | **43.6** | ✅ Best overall |
| Polynomial Regression (deg 2) | 0.47 | 43.9 | Mild nonlinearity gain |
| Decision Tree (depth=3) | 0.46 | 44.0 | Slightly underfits |
| Decision Tree (depth=6) | 0.34 | 49.1 | Overfits training data |
| kNN (k=3) | 0.43 | 45.2 | High variance |
| kNN (k=7) | 0.48 | 44.1 | Best nonlinear generalization |

**Final Test Set Results (Best Model):**
- **R²:** 0.51  
- **MAE:** 42.85  
- **MAPE:** 33.1%

---

## 🧮 Best Model Equation

\[
\hat{y} = 152.13 
- 24.87(\text{age})
- 16.38(\text{sex})
+ 519.84(\text{bmi})
+ 327.56(\text{bp})
- 186.69(\text{s1})
+ 115.72(\text{s2})
- 161.04(\text{s3})
+ 89.02(\text{s4})
+ 522.69(\text{s5})
+ 64.78(\text{s6})
\]

---

## 🔍 Key Findings

- **BMI**, **S5**, and **BP** are the strongest positive predictors.  
- Linear relationships dominate; higher-degree polynomials add little gain.  
- Decision Trees and kNN perform comparably but risk overfitting.  
- Logistic Regression (after binning target) achieves ~75% classification accuracy but isn’t appropriate for continuous regression.

---

## ⚠️ Model Limitations

- Linear model cannot capture **nonlinear or interactive effects**.  
- Moderate **R² (~0.5)** indicates ~50% of disease progression remains unexplained.  
- **Multicollinearity** among serum features (S1–S6) may affect coefficient stability.  
- Dataset is small (n=442) → limited generalization.  
- Predictions are **biased toward the mean**, struggling with extreme progression cases.

---

## 🚀 Future Work

- Add **interaction terms** and **nonlinear models** (Random Forest, XGBoost).  
- Explore **regularization** (Ridge, Lasso) to handle multicollinearity.  
- Perform **cross-validation** for more robust metrics.  
- Incorporate **additional clinical features** (diet, medication, exercise) for better accuracy.

---

## 🧰 Requirements

Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
