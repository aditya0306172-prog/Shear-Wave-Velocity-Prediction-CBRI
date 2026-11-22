import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import GroupKFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, 
                             mean_absolute_percentage_error, explained_variance_score, max_error)
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, 
                              HistGradientBoostingRegressor, GradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# NiaPy imports for Grey Wolf Optimizer
from niapy.algorithms.basic import GreyWolfOptimizer
from niapy.problems import Problem
from niapy.task import Task

# ---------------------------------------------------------
# SETUP & PATHS
# ---------------------------------------------------------
DATA_PATH = "clean_kiknet_dataset_physics.csv"
FIG_DIR = "paper_figures"
RESULTS_FILE = "model_results_summary_physics.md"

os.makedirs(FIG_DIR, exist_ok=True)

# Formatting settings
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# ---------------------------------------------------------
# PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------
def run_eda(df):
    print("[STATUS] Running Exploratory Data Analysis...")
    
    # 1. Depth vs Vs Distribution & Density
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df, x='Vs_m_s', y='Depth_m', alpha=0.3, color='indigo')
    plt.gca().invert_yaxis()
    plt.title('Depth vs Shear Wave Velocity (Vs)')
    plt.xlabel('Shear Wave Velocity (Vs) [m/s]')
    plt.ylabel('Depth [m]')
    plt.savefig(os.path.join(FIG_DIR, "eda_vs_depth_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr_features = ['Depth_m', 'N_Value', 'Vp_m_s', 'Density_g_cm3', 'FC', 'qc_MPa', 'Sigma_v_eff_kPa', 'Void_Ratio_e', 'Vs_m_s']
    corr_matrix = df[corr_features].corr(method='spearman')
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Spearman Correlation Matrix (Including Physics Features)')
    plt.savefig(os.path.join(FIG_DIR, "eda_correlation_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Geology vs Vs Boxplot
    plt.figure(figsize=(12, 6))
    order = df.groupby('Lithology')['Vs_m_s'].median().sort_values().index
    sns.boxplot(data=df, x='Lithology', y='Vs_m_s', order=order, palette='viridis', hue='Lithology', legend=False)
    plt.xticks(rotation=45, ha='right')
    plt.title('Shear Wave Velocity (Vs) by Soil Lithology')
    plt.ylabel('Vs [m/s]')
    plt.savefig(os.path.join(FIG_DIR, "eda_geology_vs_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# PHASE 3: MODEL BUILDING & EXPERIMENTATION
# ---------------------------------------------------------

class XGBoostTuningProblem(Problem):
    def __init__(self, X_train, y_train, groups_train):
        super().__init__(dimension=5, lower=0.0, upper=1.0)
        self.X_train = X_train
        self.y_train = y_train
        self.groups_train = groups_train

    def _evaluate(self, x):
        n_estimators = int(x[0] * 400) + 100
        max_depth = int(x[1] * 7) + 3
        lr = 10 ** (x[2] * (np.log10(0.3) - np.log10(1e-3)) + np.log10(1e-3))
        subsample = x[3] * 0.4 + 0.6
        colsample_bytree = x[4] * 0.4 + 0.6
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=42
        )
        
        gkf = GroupKFold(n_splits=3)
        scores = []
        for train_idx, val_idx in gkf.split(self.X_train, self.y_train, self.groups_train):
            X_tr, X_va = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_va = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            model.fit(X_tr, y_tr, verbose=False)
            preds = model.predict(X_va)
            scores.append(mean_squared_error(y_va, preds))
        return np.mean(scores)

class LightGBMTuningProblem(Problem):
    def __init__(self, X_train, y_train, groups_train):
        super().__init__(dimension=5, lower=0.0, upper=1.0)
        self.X_train = X_train
        self.y_train = y_train
        self.groups_train = groups_train

    def _evaluate(self, x):
        n_estimators = int(x[0] * 400) + 100
        max_depth = int(x[1] * 7) + 3
        lr = 10 ** (x[2] * (np.log10(0.3) - np.log10(1e-3)) + np.log10(1e-3))
        subsample = x[3] * 0.4 + 0.6
        colsample_bytree = x[4] * 0.4 + 0.6
        
        model = lgb.LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=lr, subsample=subsample,
            colsample_bytree=colsample_bytree, random_state=42, verbose=-1
        )
        
        gkf = GroupKFold(n_splits=3)
        scores = []
        for train_idx, val_idx in gkf.split(self.X_train, self.y_train, self.groups_train):
            X_tr, X_va = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_va = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            scores.append(mean_squared_error(y_va, preds))
        return np.mean(scores)


def main():
    df = pd.read_csv(DATA_PATH)
    run_eda(df)
    
    print("[STATUS] Data Preprocessing...")
    target = 'Vs_m_s'
    features = ['Depth_m', 'N_Value', 'Vp_m_s', 'Density_g_cm3', 'Lithology', 'FC', 'qc_MPa', 'Sigma_v_kPa', 'Sigma_v_eff_kPa', 'Void_Ratio_e']
    groups = df['Station_ID']
    
    X = df[features].copy()
    y = df[target].copy()
    
    # 1. Target Encoding for Lithology
    encoder = TargetEncoder(target_type="continuous")
    X['Lithology'] = encoder.fit_transform(X[['Lithology']], y)
    
    # 2. KNN Imputation
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Validation Strategy: 1 Hold-out test set for final eval (Grouped)
    gkf_outer = GroupKFold(n_splits=5)
    train_idx, test_idx = next(gkf_outer.split(X_imputed, y, groups))
    
    X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    
    # Scale for distance/neural models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("[STATUS] Running Hyperparameter Tuning (Grey Wolf Optimizer)...")
    
    xgb_task = Task(problem=XGBoostTuningProblem(X_train, y_train, groups_train), max_iters=3)
    algo_xgb = GreyWolfOptimizer(population_size=4)
    best_xgb_x, _ = algo_xgb.run(xgb_task)
    best_xgb_params = {
        'n_estimators': int(best_xgb_x[0] * 400) + 100,
        'max_depth': int(best_xgb_x[1] * 7) + 3,
        'learning_rate': 10 ** (best_xgb_x[2] * (np.log10(0.3) - np.log10(1e-3)) + np.log10(1e-3)),
        'subsample': best_xgb_x[3] * 0.4 + 0.6,
        'colsample_bytree': best_xgb_x[4] * 0.4 + 0.6
    }
    
    lgb_task = Task(problem=LightGBMTuningProblem(X_train, y_train, groups_train), max_iters=3)
    algo_lgb = GreyWolfOptimizer(population_size=4)
    best_lgb_x, _ = algo_lgb.run(lgb_task)
    best_lgb_params = {
        'n_estimators': int(best_lgb_x[0] * 400) + 100,
        'max_depth': int(best_lgb_x[1] * 7) + 3,
        'learning_rate': 10 ** (best_lgb_x[2] * (np.log10(0.3) - np.log10(1e-3)) + np.log10(1e-3)),
        'subsample': best_lgb_x[3] * 0.4 + 0.6,
        'colsample_bytree': best_lgb_x[4] * 0.4 + 0.6
    }
    
    print("[STATUS] Training Final Models...")
    models = {
        "XGBoost": xgb.XGBRegressor(**best_xgb_params, random_state=42),
        "LightGBM": lgb.LGBMRegressor(**best_lgb_params, random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(verbose=0, random_seed=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR (RBF)": SVR(kernel='rbf'),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance'),
        "MLPRegressor": MLPRegressor(random_state=42, max_iter=500),
        "Ridge": Ridge(alpha=1.0)
    }
    
    scale_models = ["Ridge", "SVR (RBF)", "KNN", "MLPRegressor"]
    results = []
    trained_models = {}
    
    n = len(y_test)
    p = X_test.shape[1]
    
    for name, model in models.items():
        if name in scale_models:
            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        r2 = r2_score(y_test, preds)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds) * 100
        max_err = max_error(y_test, preds)
        exp_var = explained_variance_score(y_test, preds)
        
        results.append({
            "Model": name,
            "R2_val": r2,
            "Adj_R2_val": adj_r2,
            "RMSE_val": rmse,
            "MAE_val": mae,
            "MAPE_val": mape,
            "MaxError_val": max_err,
            "ExpVar_val": exp_var
        })
        trained_models[name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2_val", ascending=False).reset_index(drop=True)
    
    # Identify Best Model
    best_model_name = results_df.iloc[0]["Model"]
    results_df["Status"] = ""
    results_df.loc[0, "Status"] = "**BEST MODEL**"
    
    # ---------------------------------------------------------
    # Generate Model Ranking Bar Chart
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="R2_val", y="Model", hue="Model", palette="viridis", legend=False)
    plt.title("Model Ranking by R² Score")
    plt.xlabel("R² Score")
    plt.ylabel("Model")
    plt.savefig(os.path.join(FIG_DIR, "model_ranking_bar_chart.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Format the dataframe for markdown export
    final_df = pd.DataFrame()
    final_df["Model"] = results_df["Model"]
    final_df["R² ↑"] = results_df["R2_val"].apply(lambda x: f"{x:.4f}")
    final_df["Adj. R² ↑"] = results_df["Adj_R2_val"].apply(lambda x: f"{x:.4f}")
    final_df["RMSE (m/s) ↓"] = results_df["RMSE_val"].apply(lambda x: f"{x:.2f}")
    final_df["MAE (m/s) ↓"] = results_df["MAE_val"].apply(lambda x: f"{x:.2f}")
    final_df["MAPE (%) ↓"] = results_df["MAPE_val"].apply(lambda x: f"{x:.2f}")
    final_df["Max Error (m/s) ↓"] = results_df["MaxError_val"].apply(lambda x: f"{x:.2f}")
    final_df["Explained Var ↑"] = results_df["ExpVar_val"].apply(lambda x: f"{x:.4f}")
    final_df["Status"] = results_df["Status"]

    # Save markdown results
    with open(RESULTS_FILE, 'w') as f:
        f.write("# Physics-Informed Model Results Summary\n\n")
        f.write("Performance evaluation of various machine learning models predicting Shear Wave Velocity (Vs), optimized via Grey Wolf Optimizer (GWO).\n\n")
        f.write(final_df.to_markdown(index=False))
        
    print(f"[STATUS] Results saved to {RESULTS_FILE}")
    print(f"[STATUS] Best Model Identified: {best_model_name}")
    
    # ---------------------------------------------------------
    # PHASE 4: EVALUATION & RESEARCH DIAGRAMS
    # ---------------------------------------------------------
    print("[STATUS] Generating Evaluation Plots...")
    best_model = trained_models[best_model_name]
    
    if best_model_name in scale_models:
        final_preds = best_model.predict(X_test_scaled)
    else:
        final_preds = best_model.predict(X_test)
    
    # 1. Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, final_preds, alpha=0.3, color='teal')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.title(f'Actual vs Predicted Vs ({best_model_name})')
    plt.xlabel('Actual Vs [m/s]')
    plt.ylabel('Predicted Vs [m/s]')
    plt.savefig(os.path.join(FIG_DIR, "actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[STATUS] ML Pipeline Execution Completed!")

if __name__ == "__main__":
    main()
