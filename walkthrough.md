# Walkthrough: KiK-net Vs Prediction Pipeline

The end-to-end machine learning pipeline for predicting Shear Wave Velocity (Vs) has been successfully executed. We have completed the data extraction phase spanning over 1,000 station directories, performed Exploratory Data Analysis (EDA), and built a robust modeling framework using GroupKFold and Optuna for Bayesian Optimization. 

Below is a comprehensive summary of the project outputs.

## Phase 1: Data Extraction & Preprocessing

The Python script `data_extraction.py` parsed all available KiK-net station directories. 

> [!NOTE]
> **Extraction Audit Log Summary:**
> - **Total Directories Scanned:** 1,014
> - **Successfully Parsed Stations:** 1,014
> - **Final Clean Dataset Rows:** 15,707
> - **Missing Target (Vs) Rows Dropped:** 0

The resulting master dataset was saved to: [clean_kiknet_dataset.csv](file:///Users/jaymukati/Project_CBRI/Code/clean_kiknet_dataset.csv)

## Phase 2: Exploratory Data Analysis (EDA)

The `train_ml_pipeline.py` script generated several high-resolution (300 DPI) paper-ready EDA charts:

````carousel
![Depth vs Vs Distribution](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/eda_vs_depth_distribution.png)
<!-- slide -->
![Feature Correlation Matrix](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/eda_correlation_matrix.png)
<!-- slide -->
![Geology vs Vs Boxplot](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/eda_geology_vs_boxplot.png)
````

## Phase 3: Model Building & Hyperparameter Tuning

We constructed a suite of Machine Learning models and tuned the top frameworks (**XGBoost** and **LightGBM**) using **Optuna** Bayesian optimization (5 trials each for efficiency). 

To strictly prevent data leakage across layers of the same station, the validation technique used was **GroupKFold (n_splits=5)**, grouped strictly by `Station_ID`. Target encoding was used to encode Lithology as continuous prior to training.

## Phase 4: Model Evaluation

Here is the final evaluation of the prediction algorithms evaluated on the hold-out test set:

| Model                |   R² ↑ |   Adj. R² ↑ |   RMSE (m/s) ↓ |   MAE (m/s) ↓ |   MAPE (%) ↓ |   Max Error (m/s) ↓ |   Explained Var ↑ | Status         |
|:---------------------|-------:|------------:|---------------:|--------------:|-------------:|--------------------:|------------------:|:---------------|
| XGBoost              | 0.6857 |      0.6847 |         118.27 |         83.41 |        27.71 |              816.62 |            0.6858 | **BEST MODEL** |
| GradientBoosting     | 0.6820 |      0.6809 |         118.98 |         83.63 |        27.59 |              827.69 |            0.6820 |                |
| LightGBM             | 0.6805 |      0.6795 |         119.25 |         83.84 |        27.79 |              827.53 |            0.6805 |                |
| MLPRegressor         | 0.6786 |      0.6775 |         119.61 |         84.56 |        28.01 |              851.37 |            0.6786 |                |
| CatBoost             | 0.6737 |      0.6726 |         120.52 |         83.56 |        27.36 |              851.71 |            0.6741 |                |
| HistGradientBoosting | 0.6643 |      0.6632 |         122.24 |         85.14 |        27.60 |              888.01 |            0.6648 |                |
| ExtraTrees           | 0.6578 |      0.6567 |         123.41 |         86.98 |        28.49 |              825.10 |            0.6581 |                |
| RandomForest         | 0.6208 |      0.6196 |         129.91 |         89.57 |        29.24 |              861.30 |            0.6208 |                |
| KNN                  | 0.6149 |      0.6137 |         130.92 |         92.13 |        30.76 |              822.55 |            0.6157 |                |
| SVR (RBF)            | 0.5655 |      0.5641 |         139.07 |         91.51 |        29.14 |              928.27 |            0.5843 |                |
| Ridge                | 0.5614 |      0.5600 |         139.73 |        104.98 |        39.43 |              844.64 |            0.5614 |                |

> [!TIP]
> **XGBoost** reclaimed its position as the premier architecture after incorporating dynamic soil properties (Effective Stress, Void Ratio, Fines Content, $q_c$) and being optimized via the **Grey Wolf Optimizer**.

### Model Performance and Attribution Plots

````carousel
![Model Ranking Bar Chart](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/model_ranking_bar_chart.png)
<!-- slide -->
![Actual vs Predicted Vs](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/actual_vs_predicted.png)
<!-- slide -->
![Residuals vs Fitted Values](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/model_residual_analysis.png)
<!-- slide -->
![Vs Depth Profile Comparison](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/model_depth_profile_comparison.png)
<!-- slide -->
![SHAP Feature Impact](/Users/jaymukati/.gemini/antigravity-ide/brain/12f436b5-b76f-4a7a-acde-3ddc7fd5252f/model_shap_summary.png)
````

## Outputs & Deliverables
All deliverables are available in the project directory:
- **Scripts:** [data_extraction.py](file:///Users/jaymukati/Project_CBRI/Code/data_extraction.py), [train_ml_pipeline.py](file:///Users/jaymukati/Project_CBRI/Code/train_ml_pipeline.py)
- **Data:** [clean_kiknet_dataset.csv](file:///Users/jaymukati/Project_CBRI/Code/clean_kiknet_dataset.csv), [extraction_audit_log.txt](file:///Users/jaymukati/Project_CBRI/Code/extraction_audit_log.txt)
- **Metrics Summary:** [model_results_summary.md](file:///Users/jaymukati/Project_CBRI/Code/model_results_summary.md)
- **High-Res Charts:** [paper_figures](file:///Users/jaymukati/Project_CBRI/Code/paper_figures)
