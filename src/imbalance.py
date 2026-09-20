"""
Class Imbalance Handling Module.
Implements and benchmarks the 7 resampling methods evaluated in Section 3.4 and Table 5 of Akinjole et al. (2024):
1. None (Imbalanced)
2. Random Over-Sampling (ROS)
3. Random Under-Sampling (RUS)
4. SMOTE
5. ADASYN
6. Tomek Links
7. SMOTE-Tomek
8. SMOTE + ENN (Paper Winner)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier


def get_resampler(method: str, random_state: int = 42):
    """
    Returns the specified resampler instance.
    """
    key = "".join(c for c in method.lower() if c.isalnum())
    
    if key in ["none", "baseline"]:
        return None
    elif key in ["ros", "randomoversampler", "randomoversampling"]:
        return RandomOverSampler(random_state=random_state)
    elif key in ["rus", "randomundersampler", "randomundersampling"]:
        return RandomUnderSampler(random_state=random_state)
    elif key == "smote":
        return SMOTE(random_state=random_state)
    elif key == "adasyn":
        return ADASYN(random_state=random_state)
    elif key in ["tomek", "tomeklinks"]:
        return TomekLinks()
    elif key in ["smotetomek", "smotetomeklinks"]:
        return SMOTETomek(random_state=random_state)
    elif key in ["smoteenn", "smoteplusenn", "smoteandenn"]:
        return SMOTEENN(random_state=random_state)
    else:
        raise ValueError(f"Unknown resampling method: {method}")


def apply_resampling(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote_enn",
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies the chosen resampling strategy to the training dataset.
    Default is SMOTE+ENN, the best-performing method identified in the paper.
    """
    resampler = get_resampler(method, random_state=random_state)
    if resampler is None:
        return X, y
    
    X_res, y_res = resampler.fit_resample(X, y)
    return X_res, y_res


def benchmark_resampling_techniques(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Systematically evaluates all 8 resampling strategies using XGBoost
    to reproduce Table 5 from the paper.
    """
    methods = [
        "None",
        "ROS",
        "SMOTE",
        "ADASYN",
        "RUS",
        "Tomek-Links",
        "SMOTE-Tomek",
        "SMOTE + ENN"
    ]
    
    results = []
    print("\n--- Benchmarking Class Imbalance Resampling Methods (Table 5) ---")
    
    for m in methods:
        print(f"[*] Evaluating {m}...")
        try:
            if m == "None":
                X_tr, y_tr = X_train.values, y_train.values
            else:
                method_key = m.lower().replace("-", "_").replace(" ", "_").replace("+", "")
                X_tr, y_tr = apply_resampling(X_train.values, y_train.values, method=method_key, random_state=random_state)
            
            # Use XGBoost as the benchmark evaluator (Section 3.4 of the paper)
            clf = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1
            )
            clf.fit(X_tr, y_tr)
            
            y_pred = clf.predict(X_test.values)
            y_proba = clf.predict_proba(X_test.values)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            
            results.append({
                "Method": m,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "AUC": round(auc, 4)
            })
            print(f"    -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, AUC: {auc:.4f}")
        except Exception as e:
            print(f"    [!] Error testing {m}: {e}")
            
    df_res = pd.DataFrame(results)
    return df_res
