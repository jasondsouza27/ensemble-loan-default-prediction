"""
Feature Engineering Module for Loan Default Risk Prediction.
Implements regional mapping, quantile binning, financial interactions, and one-hot encoding
as detailed in Sections 3.2, 3.3, Table 2, and Table A3 of Akinjole et al. (2024).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import STATE_TO_REGION


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering:
    1. Maps addr_state -> region (MidWest, NorthEast, SouthEast, SouthWest, West)
    2. Cleans term -> numeric integer months (36, 60)
    3. Generates binned features: annual_inc_binned & revol_bal_binned
    4. Computes financial interaction terms: loan_amnt_dti, annual_inc_installment, annual_inc_dti
    5. One-hot encodes categorical variables per Table 2
    """
    df = df.copy()

    # 1. State to Region mapping
    if "addr_state" in df.columns:
        df["region"] = df["addr_state"].map(STATE_TO_REGION).fillna("West")
        df.drop(columns=["addr_state"], inplace=True)
    elif "region" not in df.columns:
        df["region"] = "West"

    # 2. Term cleaning
    if "term" in df.columns:
        if df["term"].dtype == object or isinstance(df["term"].iloc[0], str):
            df["term_months"] = df["term"].astype(str).str.extract(r'(\d+)').astype(float)
            df["term"] = df["term_months"]
            df.drop(columns=["term_months"], inplace=True)
        else:
            df["term"] = df["term"].astype(float)

    # 3. Binning annual_inc and revol_bal (5 categories: Very Low, Low, Medium, High, Very High)
    bin_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    
    # Use qcut with fallback to cut in case of duplicate bin edges
    try:
        df["annual_inc_binned"] = pd.qcut(df["annual_inc"], q=5, labels=bin_labels)
    except ValueError:
        df["annual_inc_binned"] = pd.cut(df["annual_inc"], bins=5, labels=bin_labels)
        
    try:
        df["revol_bal_binned"] = pd.qcut(df["revol_bal"], q=5, labels=bin_labels)
    except ValueError:
        df["revol_bal_binned"] = pd.cut(df["revol_bal"], bins=5, labels=bin_labels)

    # 4. Financial Interaction Features (per Table A3)
    # loan_amnt_dti: interaction between principal borrowed and debt burden
    df["loan_amnt_dti"] = (df["loan_amnt"] * df["dti"]) / 1000.0

    # annual_inc_installment: ability to service debt (annual income vs annual installment burden)
    if "installment" in df.columns:
        df["annual_inc_installment"] = df["annual_inc"] / ((df["installment"] * 12.0) + 1.0)
    else:
        # compute approx installment if missing
        df["annual_inc_installment"] = df["annual_inc"] / (df["loan_amnt"] * 0.3 + 1.0)

    # annual_inc_dti: ratio of income scaled by debt-to-income
    df["annual_inc_dti"] = df["annual_inc"] / (df["dti"] + 1.0)

    # 5. One-Hot Encoding for categorical features per Table 2
    categorical_cols = [
        "home_ownership", "verification_status", "purpose", "initial_list_status",
        "application_type", "region", "annual_inc_binned", "revol_bal_binned"
    ]
    
    # Filter only existing categorical columns
    cols_to_encode = [c for c in categorical_cols if c in df.columns]
    
    # Perform one-hot encoding with dummy prefix
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=float)
    
    return df_encoded
