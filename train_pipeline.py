"""
End-to-End Orchestration Pipeline for Loan Default Risk Prediction.
Executes the full pipeline:
1. Data Ingestion
2. Data Cleaning & Winsorization & Robust Scaling (Section 3.1 - 3.3)
3. Class Imbalance Handling via SMOTE + ENN (Section 3.4, Table 5)
4. Feature Selection via RFECV with Recall Scorer (Section 3.5, Figure 8)
5. Model Training: 6 Base Models + 4 Ensembles (Section 3.6, Table 4, 6, 7)
6. Evaluation and Visualization (Figures 10, 11, 12, 13)
7. SHAP Explainability (Figure 9, Table A3)
"""

import sys
import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from src.config import MODELS_DIR, PROCESSED_DATA_DIR
from data.generate_sample_data import load_or_generate_dataset
from src.data_preprocessing import prepare_data, LoanDataPreprocessor
from src.imbalance import apply_resampling, benchmark_resampling_techniques
from src.feature_selection import run_rfecv_selection
from src.train_models import train_and_save_all_models
from src.evaluate import (
    evaluate_models, plot_roc_curves, plot_metrics_comparison,
    plot_auc_comparison
)
from src.explainability import (
    get_tree_explainer, compute_shap_values, plot_global_feature_importance
)


def run_pipeline(
    data_path: str = None,
    sample_size: int = 25000,
    run_imbalance_benchmark: bool = True,
    run_rfecv: bool = True,
    fast_mode: bool = True
):
    print("=" * 70)
    print("  ENSEMBLE-BASED LOAN DEFAULT RISK PREDICTION PIPELINE")
    print("  Based on Akinjole et al. (2024), MDPI Mathematics 12, 3423")
    print("=" * 70)

    # 1. Load Data
    df_raw = load_or_generate_dataset(filepath=data_path, n_samples=sample_size)
    print(f"\n[1] Data Loaded: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Preprocess, Winsorize, RobustScale, 80:20 Train-Test Split
    print("\n[2] Executing Data Cleaning, Winsorization & Robust Scaling...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        df_raw, test_size=0.20, random_state=42
    )
    print(f"    Train size: {X_train.shape}, Test size: {X_test.shape}")
    print(f"    Train label balance: {y_train.value_counts(normalize=True).to_dict()}")

    # 3. Class Imbalance Benchmark & Resampling
    if run_imbalance_benchmark:
        print("\n[3] Running Class Imbalance Benchmark (Table 5)...")
        # Subsample slightly for fast benchmarking
        bench_sub_idx = np.random.choice(len(X_train), size=min(len(X_train), 8000), replace=False)
        test_sub_idx = np.random.choice(len(X_test), size=min(len(X_test), 3000), replace=False)
        df_imbalance_bench = benchmark_resampling_techniques(
            X_train.iloc[bench_sub_idx], y_train.iloc[bench_sub_idx],
            X_test.iloc[test_sub_idx], y_test.iloc[test_sub_idx]
        )
        print("\nBenchmark Results (Table 5 Replicated):")
        print(df_imbalance_bench.to_string(index=False))
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        df_imbalance_bench.to_csv(MODELS_DIR / "imbalance_benchmark.csv", index=False)

    print("\n[3b] Applying Winning Resampling Method: SMOTE + ENN...")
    X_train_res, y_train_res = apply_resampling(
        X_train.values, y_train.values, method="smote_enn", random_state=42
    )
    print(f"    Resampled Training Size: {X_train_res.shape[0]} samples (balanced)")

    # 4. Feature Selection via RFECV
    feature_names = list(X_train.columns)
    if run_rfecv:
        print("\n[4] Running Feature Selection (RFECV with Recall scoring)...")
        df_res_train = pd.DataFrame(X_train_res, columns=feature_names)
        # Run on subsample for speed
        rfecv_sub_idx = np.random.choice(len(df_res_train), size=min(len(df_res_train), 6000), replace=False)
        selected_features, importances, fig_rfecv = run_rfecv_selection(
            df_res_train.iloc[rfecv_sub_idx], y_train_res[rfecv_sub_idx],
            min_features_to_select=min(30, len(feature_names)),
            target_features=min(48, len(feature_names)),
            cv_folds=3,
            step=1
        )
        if fig_rfecv:
            fig_rfecv.savefig(MODELS_DIR / "rfecv_curve.png", dpi=300)
    else:
        selected_features = feature_names

    print(f"    Selected Features: {len(selected_features)}")
    
    # Filter datasets by selected features
    feat_indices = [feature_names.index(f) for f in selected_features]
    X_train_sel = X_train_res[:, feat_indices]
    X_test_sel = X_test[selected_features].values

    # 5. Train Base Models and Ensembles (Table 4, 6, 7)
    print("\n[5] Training 6 Base Models and 4 Ensembles...")
    trained_models = train_and_save_all_models(
        X_train_sel, y_train_res,
        feature_names=selected_features,
        fast_mode=fast_mode,
        random_state=42
    )

    # 6. Evaluate Models
    print("\n[6] Evaluating All Models on Test Set...")
    df_base, df_ensemble, detailed_eval = evaluate_models(
        trained_models, X_test_sel, y_test.values
    )

    print("\n=======================================================")
    print("         TABLE 6: INDIVIDUAL MODEL RESULTS             ")
    print("=======================================================")
    print(df_base.to_string(index=False))

    print("\n=======================================================")
    print("         TABLE 7: ENSEMBLE MODEL RESULTS               ")
    print("=======================================================")
    print(df_ensemble.to_string(index=False))

    df_base.to_csv(MODELS_DIR / "base_models_results.csv", index=False)
    df_ensemble.to_csv(MODELS_DIR / "ensemble_models_results.csv", index=False)

    # 7. Generate Evaluation Plots
    print("\n[7] Generating Comparison Visualizations...")
    # Individual ROC curves
    fig_roc_base = plot_roc_curves(
        detailed_eval,
        ["Random Forest", "Decision Tree", "SVM", "XGBoost", "ADABoost", "MLP"],
        title="Receiver Operating Characteristic (ROC) Curve - Individual Models",
        save_path=MODELS_DIR / "roc_individual.png"
    )
    # All ROC curves
    all_model_keys = list(trained_models.keys())
    fig_roc_all = plot_roc_curves(
        detailed_eval,
        all_model_keys,
        title="Receiver Operating Characteristic (ROC) Curve - All Models",
        save_path=MODELS_DIR / "roc_all_models.png"
    )
    # Individual model multi-metric comparison
    fig_metrics = plot_metrics_comparison(
        df_base,
        title="Individual Model Performance Comparison",
        save_path=MODELS_DIR / "metrics_comparison.png"
    )
    # All models AUC comparison
    df_all_metrics = pd.concat([df_base, df_ensemble], ignore_index=True)
    fig_auc = plot_auc_comparison(
        df_all_metrics,
        title="Model Performance Comparison (AUC) - Highlighting Stacking Champion",
        save_path=MODELS_DIR / "auc_comparison_all.png"
    )

    # 8. SHAP Explainability
    print("\n[8] Computing SHAP Values for Explainability...")
    xgb_model = trained_models.get("XGBoost")
    if xgb_model:
        X_test_sample = pd.DataFrame(X_test_sel[:250], columns=selected_features)
        explainer = get_tree_explainer(xgb_model, X_test_sample)
        shap_values = compute_shap_values(explainer, X_test_sample)
        
        fig_shap = plot_global_feature_importance(
            shap_values,
            selected_features,
            max_display=25,
            save_path=MODELS_DIR / "shap_feature_importance.png"
        )
        joblib.dump(explainer, MODELS_DIR / "shap_explainer.joblib")
        print("    -> SHAP explainer saved to models/shap_explainer.joblib")

    # 9. Save Preprocessor & Metadata
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(selected_features, MODELS_DIR / "selected_features.joblib")
    joblib.dump(detailed_eval, MODELS_DIR / "detailed_evaluation.joblib")
    
    print("\n" + "=" * 70)
    print("  PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS SAVED IN models/")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline(sample_size=20000, fast_mode=True)
