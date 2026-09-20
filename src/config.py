"""
Configuration and metadata definitions for Ensemble-Based Loan Default Risk Prediction.
Based on Akinjole et al., MDPI Mathematics 2024 (12, 3423).
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Target Definition
# 0 = "Fully Paid" (Non-default), 1 = "Charged off" (Default)
TARGET_COL = "loan_status"
TARGET_MAPPING = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1
}

# Features to exclude outright (leakage, high missing rate > 50%, or PPI per Table A1)
EXCLUDED_FEATURES = [
    "id", "member_id", "url", "desc", "title", "emp_title", "emp_length", "zip_code",
    "funded_amnt", "funded_amnt_inv", "pymnt_plan", "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
    "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "collections_12_mths_ex_med", "policy_code", "acc_now_delinq", "tot_coll_amt",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "bc_util",
    "chargeoff_within_12_mths", "mo_sin_old_il_acct", "mths_since_recent_inq",
    "num_accts_ever_120_pd", "num_il_tl", "num_rev_accts", "num_rev_tl_bal_gt_0",
    "num_sats", "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "num_tl_op_past_12m", "tax_liens", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "hardship_flag", "disbursement_method",
    "debt_settlement_flag", "issue_d_yr", "earliest_cr_line_yr", "fico_range_high"
]

# Core Predictor Features (Table A1 & Table 2)
CATEGORICAL_FEATURES = [
    "home_ownership",
    "verification_status",
    "purpose",
    "initial_list_status",
    "application_type",
    "region"
]

NUMERICAL_FEATURES = [
    "loan_amnt", "term_months", "int_rate", "installment", "annual_inc", "dti",
    "fico_range_low", "open_acc", "pub_rec", "revol_bal", "total_acc",
    "avg_cur_bal", "bc_open_to_buy", "delinq_amnt", "delinq_2yrs",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats",
    "num_bc_tl", "num_op_rev_tl", "pct_tl_nvr_dlq", "percent_bc_gt_75",
    "pub_rec_bankruptcies", "tot_hi_cred_lim"
]

# US State to Region Mapping per Section 3.2
STATE_TO_REGION = {
    # NorthEast
    "CT": "NorthEast", "ME": "NorthEast", "MA": "NorthEast", "NH": "NorthEast",
    "RI": "NorthEast", "VT": "NorthEast", "NJ": "NorthEast", "NY": "NorthEast", "PA": "NorthEast",
    # MidWest
    "IL": "MidWest", "IN": "MidWest", "MI": "MidWest", "OH": "MidWest", "WI": "MidWest",
    "IA": "MidWest", "KS": "MidWest", "MN": "MidWest", "MO": "MidWest", "NE": "MidWest",
    "ND": "MidWest", "SD": "MidWest",
    # SouthEast
    "DE": "SouthEast", "FL": "SouthEast", "GA": "SouthEast", "MD": "SouthEast",
    "NC": "SouthEast", "SC": "SouthEast", "VA": "SouthEast", "WV": "SouthEast",
    "AL": "SouthEast", "KY": "SouthEast", "MS": "SouthEast", "TN": "SouthEast", "DC": "SouthEast",
    # SouthWest
    "AR": "SouthWest", "LA": "SouthWest", "OK": "SouthWest", "TX": "SouthWest",
    "AZ": "SouthWest", "NM": "SouthWest",
    # West
    "CO": "West", "ID": "West", "MT": "West", "UT": "West", "WY": "West",
    "AK": "West", "CA": "West", "HI": "West", "NV": "West", "OR": "West", "WA": "West"
}

# Best Hyperparameters from Table 4 (Paper)
BEST_PARAMS = {
    "RandomForest": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "random_state": 42,
        "n_jobs": -1
    },
    "DecisionTree": {
        "criterion": "gini",
        "max_depth": 15,
        "min_impurity_decrease": 0.01,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "random_state": 42
    },
    "SVM": {
        "C": 1.0,
        "degree": 2,
        "gamma": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42
    },
    "XGBoost": {
        "n_estimators": 200,
        "max_depth": 20,
        "learning_rate": 0.1,
        "colsample_bytree": 0.9,
        "subsample": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 1.5,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1
    },
    "AdaBoost": {
        "n_estimators": 300,
        "learning_rate": 0.15,
        "random_state": 42
    },
    "MLP": {
        "hidden_layer_sizes": (150, 150, 150),
        "activation": "relu",
        "alpha": 0.001,
        "solver": "adam",
        "learning_rate": "constant",
        "batch_size": 256,
        "early_stopping": True,
        "random_state": 42,
        "max_iter": 300
    }
}
