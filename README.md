
# Ridge Log-Price Modeling & Anomaly Detection

This repository contains a compact, end-to-end example of **price modeling** and **residual-based anomaly detection** using a **synthetic dataset**.  
It demonstrates:

- Generating realistic synthetic pricing data with categorical and numerical features
- Training a **Ridge regression** model on **log(1 + price)**
- **Bias correction** when retransforming predictions using the **smearing estimator**
- Evaluating performance with **MAE**, **R²**, and **bootstrap confidence intervals**
- **Anomaly detection** using robust residual z-scores (MAD) and **IsolationForest**
- Saving artifacts: datasets, top anomalies, and diagnostic plots

> ⚠️ Note: The script included intentionally **injects anomalies** into `price` and `diameter` to simulate outliers for demonstration purposes.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What the Script Does](#what-the-script-does)
- [Requirements](#requirements)


---

## Project Overview

We model a synthetic pricing problem with the following attributes:

- **brand**: A, B, C, D  
- **age_years**: continuous (gamma distribution, clipped)
- **condition**: poor, good, like_new
- **diameter**: continuous (normal distribution, clipped)
- **has_box**, **has_papers**: binary indicators

The target variable is **price**, generated from a ground-truth data-generating process (DGP) in **log space** with **heteroscedastic noise** (variance grows with age). Some deliberate **anomalies** are injected to test detection logic.

---

## What the Script Does

1. **Data Generation (`make_data`)**
   - Creates a DataFrame with features listed above
   - Computes **log-price** based on brand/condition effects, age, diameter, and accessories
   - Adds **heteroscedastic noise**, then transforms to `price = exp(log_price)`
   - Injects **outliers** into `price` and `diameter`

2. **Modeling (`build_model` + training in `main`)**
   - Preprocesses categorical features using **OneHotEncoder**
   - Scales numeric features using **StandardScaler**
   - Trains a **Ridge Regression** model (`alpha=1.0`, `solver="sag"`) on **log(1 + price)**

3. **Bias Correction (Smearing Estimator)**
   - Corrects retransform bias via **smearing factor**:  
     `pred_price = expm1(pred_log) * mean(exp(residual_log))`

4. **Evaluation**
   - Computes **MAE** and **R²** on the holdout set
   - Estimates **95% bootstrap confidence intervals** for MAE and R²
   - Runs **5-fold cross-validation** on the log target (MAE)

5. **Anomaly Detection**
   - Uses **robust z-score** (via MAD) on residuals
   - Applies **IsolationForest** (contamination = 5%) on `[price_pred, resid]`
   - Combines both signals into a single **anomaly score** and saves **top 20** entries

6. **Artifact Generation**
   - Saves datasets and plots to `./artifacts_python_demo/`

---

## Requirements

- Python 3.9+ (3.10+ recommended)
- Packages:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
``

