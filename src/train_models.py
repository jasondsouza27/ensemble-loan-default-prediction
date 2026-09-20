"""
Model Training and Ensembling Module for Loan Default Risk Prediction.
Implements the 6 base models and 4 ensemble architectures (Voting A/B, Stacking A/B)
per Section 3.6, Table 4, Table 6, and Table 7 of Akinjole et al. (2024).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import BEST_PARAMS, MODELS_DIR


def get_base_models(random_state: int = 42, fast_mode: bool = False) -> Dict[str, Any]:
    """
    Initializes the 6 base classifiers with the optimal hyperparameters from Table 4.
    If fast_mode is enabled, balances parameters for rapid interactive training.
    """
    rf_params = BEST_PARAMS["RandomForest"].copy()
    dt_params = BEST_PARAMS["DecisionTree"].copy()
    svm_params = BEST_PARAMS["SVM"].copy()
    xgb_params = BEST_PARAMS["XGBoost"].copy()
    ada_params = BEST_PARAMS["AdaBoost"].copy()
    mlp_params = BEST_PARAMS["MLP"].copy()

    if fast_mode:
        rf_params["n_estimators"] = 100
        rf_params["max_depth"] = 12
        xgb_params["n_estimators"] = 100
        xgb_params["max_depth"] = 8
        ada_params["n_estimators"] = 100
        mlp_params["hidden_layer_sizes"] = (100, 100)
        mlp_params["max_iter"] = 150

    models = {
        "Random Forest": RandomForestClassifier(**rf_params),
        "Decision Tree": DecisionTreeClassifier(**dt_params),
        # SVM with probability=True for soft voting and probability stacking
        "SVM": SVC(
            C=svm_params["C"],
            kernel=svm_params["kernel"],
            gamma=svm_params["gamma"],
            probability=True,
            random_state=random_state,
            max_iter=3000 if fast_mode else 8000
        ),
        "XGBoost": XGBClassifier(**xgb_params),
        "ADABoost": AdaBoostClassifier(**ada_params),
        "MLP": MLPClassifier(**mlp_params)
    }
    return models


def build_ensemble_models(base_models: Dict[str, Any], random_state: int = 42) -> Dict[str, Any]:
    """
    Builds the 4 ensemble architectures described in Section 3.6.3 and Table 7:
    - Voting A: Soft Voting using all 6 base models
    - Voting B: Soft Voting using top 3 models (Random Forest, XGBoost, MLP)
    - Stacking A: Stacking using all 6 base models with Logistic Regression meta-learner (Proposed Champion)
    - Stacking B: Stacking using top 3 base models with Logistic Regression meta-learner
    """
    all_estimators = [(name, model) for name, model in base_models.items()]
    top3_names = ["Random Forest", "XGBoost", "MLP"]
    top3_estimators = [(name, base_models[name]) for name in top3_names if name in base_models]

    # Meta-learner for Stacking: Logistic Regression per Section 3.6.3
    meta_learner = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)

    ensembles = {
        "Voting A": VotingClassifier(
            estimators=all_estimators,
            voting="soft",
            n_jobs=-1
        ),
        "Voting B": VotingClassifier(
            estimators=top3_estimators,
            voting="soft",
            n_jobs=-1
        ),
        "Stacking A": StackingClassifier(
            estimators=all_estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1
        ),
        "Stacking B": StackingClassifier(
            estimators=top3_estimators,
            final_estimator=meta_learner,
            cv=3,
            stack_method="predict_proba",
            n_jobs=-1
        )
    }
    return ensembles


def train_and_save_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    fast_mode: bool = False,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains all individual base models and ensemble models, and persists them.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trained_models = {}

    print("\n=======================================================")
    print("      TRAINING BASE CLASSIFIERS (TABLE 4 & TABLE 6)     ")
    print("=======================================================")
    
    # 1. Base Classifiers
    base_models = get_base_models(random_state=random_state, fast_mode=fast_mode)
    
    # For SVM, if training set is large, use a representative subset to avoid OOM / indefinite hang
    max_svm_samples = 4000 if fast_mode else 10000
    
    for name, model in base_models.items():
        print(f"[+] Fitting {name}...")
        if name == "SVM" and len(X_train) > max_svm_samples:
            sub_idx = np.random.RandomState(random_state).choice(len(X_train), size=max_svm_samples, replace=False)
            model.fit(X_train[sub_idx], y_train[sub_idx])
        else:
            model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"    -> {name} trained successfully.")

    print("\n=======================================================")
    print("   TRAINING ENSEMBLES: VOTING & STACKING (TABLE 7)      ")
    print("=======================================================")
    
    ensemble_templates = build_ensemble_models(base_models, random_state=random_state)
    for name, ensemble in ensemble_templates.items():
        print(f"[+] Fitting Ensemble: {name}...")
        ensemble.fit(X_train, y_train)
        trained_models[name] = ensemble
        print(f"    -> {name} trained successfully.")

    # Persist bundle
    bundle_path = MODELS_DIR / "trained_models_bundle.joblib"
    joblib.dump({
        "models": trained_models,
        "feature_names": feature_names,
        "best_model_name": "Stacking A"
    }, bundle_path)
    print(f"\n[+] Successfully saved trained models bundle to {bundle_path}")

    return trained_models
