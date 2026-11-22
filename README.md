# Machine Learning-Based Shear Wave Velocity (Vs) Prediction

### Geotechnical Engineering | Machine Learning | Predictive Modelling

## 1. Project Overview

Shear Wave Velocity (Vs) is an important geotechnical parameter used in understanding soil behaviour, evaluating site conditions, and assessing seismic response.

This project develops a machine learning-based framework to predict shear wave velocity using geotechnical and soil-related parameters. It uses data extracted from KiK-net station records and applies multiple machine learning algorithms to identify effective predictive models.

The objective is to investigate how data-driven approaches can support geotechnical characterization and provide reliable estimates of shear wave velocity.

## 2. Objectives

- Develop a data-driven approach for predicting shear wave velocity.
- Extract and preprocess geotechnical information from KiK-net station records.
- Perform exploratory data analysis to understand relationships between soil properties and shear wave velocity.
- Implement and compare multiple machine learning regression algorithms.
- Apply appropriate validation techniques to reduce data leakage.
- Evaluate model performance using standard regression metrics.

## 3. Dataset

The project uses data extracted from KiK-net station directories.

Reported data extraction results:

| Parameter | Result |
|---|---:|
| Station directories scanned | 1,014 |
| Successfully parsed stations | 1,014 |
| Clean dataset records | 15,707 |
| Missing target values dropped | 0 |

The dataset contains geotechnical information used for model development and prediction.

## 4. Methodology

The project follows an end-to-end machine learning workflow.

**Phase 1: Data Extraction**

- Parsing KiK-net station directories.
- Extracting relevant soil and geotechnical information.
- Creating a consolidated dataset.
- Maintaining an extraction audit log.

**Phase 2: Data Preprocessing and Feature Engineering**

- Preparing the extracted dataset.
- Handling and transforming relevant features.
- Incorporating geotechnical parameters.
- Preparing features for machine learning models.

**Phase 3: Exploratory Data Analysis**

- Analysing relationships between soil depth and shear wave velocity.
- Examining feature correlations.
- Investigating variations across geological categories.

**Phase 4: Model Development**

The framework evaluates multiple regression algorithms:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting
- Extra Trees
- Multi-Layer Perceptron (MLP)
- K-Nearest Neighbours (KNN)
- Support Vector Regression (SVR)
- Ridge Regression
- HistGradientBoosting

**Phase 5: Hyperparameter Optimization and Validation**

- Optuna-based Bayesian hyperparameter optimization.
- GroupKFold cross-validation with five splits.
- Grouping by Station_ID to reduce data leakage between records from the same station.
- Target encoding for lithology.

## 5. Model Performance

The following are the reported hold-out test results from the project's walkthrough.

| Algorithm | R² | RMSE (m/s) | MAE (m/s) |
|---|---:|---:|---:|
| XGBoost | 0.6857 | 118.27 | 83.41 |
| Gradient Boosting | 0.6820 | 118.98 | 83.63 |
| LightGBM | 0.6805 | 119.25 | 83.84 |
| MLP Regressor | 0.6786 | 119.61 | 84.56 |
| CatBoost | 0.6737 | 120.52 | 83.56 |
| Random Forest | 0.6208 | 129.91 | 89.57 |
| KNN | 0.6149 | 130.92 | 92.13 |
| SVR | 0.5655 | 139.07 | 91.51 |
| Ridge Regression | 0.5614 | 139.73 | 104.98 |

**Reported best-performing model: XGBoost**

- R²: 0.6857
- RMSE: 118.27 m/s
- MAE: 83.41 m/s

These results represent the reported experimental evaluation and should be independently reproduced before being treated as verified performance.

## 6. Evaluation Metrics

The models are evaluated using:

- **R²:** Measures the proportion of variance explained by the model.
- **Adjusted R²:** Accounts for the number of predictors.
- **RMSE:** Measures the square root of average squared prediction errors.
- **MAE:** Measures average absolute prediction errors.
- **MAPE:** Measures average percentage prediction errors.
- **Explained Variance:** Measures the proportion of variance captured by predictions.

## 7. Repository Structure

```text
Shear-Wave-Velocity-Prediction-CBRI/
│
├── Code/
│   ├── knet/
│   ├── data_extraction.py
│   ├── feature_engineering.py
│   ├── train_ml_pipeline.py
│   ├── extraction_audit_log.txt
│   ├── model_results_summary.md
│   └── model_results_summary_physics.md
│
├── RPs/
│   └── Research papers
│
├── .gitignore
├── walkthrough.md
└── README.md
```

## 8. Technologies and Libraries

The project uses Python-based machine learning and data analysis tools.

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- Matplotlib
- Seaborn
- SHAP

## 9. Future Scope

Potential research extensions include:

- Improving prediction accuracy through advanced ensemble learning.
- Applying explainable AI techniques to understand feature contributions.
- Investigating model generalization across different geological regions.
- Exploring the influence of individual geotechnical parameters on shear wave velocity.
- Evaluating model robustness under different soil conditions.
- Investigating physics-informed machine learning approaches.

## 10. Research Significance

This project explores the application of machine learning to a practical geotechnical engineering problem. By combining geotechnical data processing, predictive modelling, model validation, and performance comparison, it provides a foundation for further research into data-driven soil characterization.

## 11. Acknowledgements and Attribution

This repository is based on the original project:

[Shear-Wave-Velocity-Prediction-CBRI](https://github.com/jayxmukati/Shear-Wave-Velocity-Prediction-CBRI)

Original author and repository contributors are acknowledged for the existing project structure, code, research materials, and methodology.

Individual contributions should be documented separately and accurately.

---

**Project Domain:** Machine Learning | Geotechnical Engineering | Regression | Predictive Analytics

**Repository:** Shear-Wave-Velocity-Prediction-CBRI
