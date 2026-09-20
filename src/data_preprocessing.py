"""
Data Preprocessing Pipeline for Loan Default Risk Prediction.
Implements missing value imputation (median for skewed, mode for multimodal),
multicollinearity filtering (Pearson r > 0.90), Winsorization for outlier mitigation,
and RobustScaler normalization per Sections 3.1 - 3.3 and Table 3 of Akinjole et al. (2024).
"""

import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import EXCLUDED_FEATURES, TARGET_COL, TARGET_MAPPING, PROCESSED_DATA_DIR
from src.feature_engineering import engineer_features


class LoanDataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible preprocessor replicating the paper's preprocessing methodology:
    - Missing value imputation based on skewness (median vs mode)
    - Multicollinearity removal (r > 0.90)
    - Feature engineering (regional, binning, interaction terms)
    - Winsorization for outlier handling
    - RobustScaler for normalization
    """

    def __init__(
        self,
        winsorize_limits: Tuple[float, float] = (0.01, 0.01),
        corr_threshold: float = 0.90,
        scale_features: bool = True
    ):
        self.winsorize_limits = winsorize_limits
        self.corr_threshold = corr_threshold
        self.scale_features = scale_features
        
        # Learned statistics during fit
        self.imputation_values_: Dict[str, float] = {}
        self.collinear_cols_to_drop_: List[str] = []
        self.winsorize_bounds_: Dict[str, Tuple[float, float]] = {}
        self.feature_names_: List[str] = []
        self.scaler_: Optional[RobustScaler] = None
        self.fitted_columns_: List[str] = []

    def _impute_missing(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
            if is_train:
                # If numeric, check skewness
                if pd.api.types.is_numeric_dtype(df[col]):
                    val_skew = skew(df[col].dropna())
                    if abs(val_skew) > 0.5:
                        fill_val = df[col].median()
                    else:
                        fill_val = df[col].mean()
                else:
                    fill_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                self.imputation_values_[col] = fill_val
            
            fill_val = self.imputation_values_.get(col, 0)
            df[col] = df[col].fillna(fill_val)
        return df

    def _handle_outliers_winsorize(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if is_train:
                lower = df[col].quantile(self.winsorize_limits[0])
                upper = df[col].quantile(1.0 - self.winsorize_limits[1])
                self.winsorize_bounds_[col] = (lower, upper)
            
            lower, upper = self.winsorize_bounds_.get(col, (df[col].min(), df[col].max()))
            df[col] = np.clip(df[col], lower, upper)
            
        return df

    def _remove_multicollinearity(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        if is_train:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]
            self.collinear_cols_to_drop_ = to_drop
            print(f"[Preprocessor] Multicollinearity check (r > {self.corr_threshold}): dropping {len(to_drop)} features: {to_drop}")

        df.drop(columns=[c for c in self.collinear_cols_to_drop_ if c in df.columns], inplace=True, errors="ignore")
        return df

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        
        # 1. Drop post-loan / excluded features
        cols_to_drop = [c for c in EXCLUDED_FEATURES if c in X.columns]
        if TARGET_COL in X.columns:
            cols_to_drop.append(TARGET_COL)
        X.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        
        # 2. Impute missing values
        X = self._impute_missing(X, is_train=True)
        
        # 3. Handle outliers using Winsorization before feature creation
        X = self._handle_outliers_winsorize(X, is_train=True)
        
        # 4. Feature engineering (state->region, binning, interaction terms, dummy encoding)
        X = engineer_features(X)
        
        # 5. Multicollinearity filtering
        X = self._remove_multicollinearity(X, is_train=True)
        
        # Ensure all columns are numeric
        X = X.select_dtypes(include=[np.number])
        self.fitted_columns_ = list(X.columns)
        
        # 6. Scaling with RobustScaler (Paper's winning scaler)
        if self.scale_features:
            self.scaler_ = RobustScaler()
            self.scaler_.fit(X)
            
        self.feature_names_ = self.fitted_columns_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        
        # 1. Drop excluded features
        cols_to_drop = [c for c in EXCLUDED_FEATURES if c in X.columns]
        if TARGET_COL in X.columns:
            cols_to_drop.append(TARGET_COL)
        X.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        
        # 2. Impute missing
        X = self._impute_missing(X, is_train=False)
        
        # 3. Winsorize
        X = self._handle_outliers_winsorize(X, is_train=False)
        
        # 4. Feature engineering
        X = engineer_features(X)
        
        # 5. Drop collinear columns identified in fit
        X.drop(columns=[c for c in self.collinear_cols_to_drop_ if c in X.columns], inplace=True, errors="ignore")
        
        # Align columns with training set (add missing dummies with 0, drop unseen)
        for col in self.fitted_columns_:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.fitted_columns_]
        
        # 6. Scaling
        if self.scale_features and self.scaler_ is not None:
            scaled_vals = self.scaler_.transform(X)
            X = pd.DataFrame(scaled_vals, columns=self.fitted_columns_, index=X.index)
            
        return X

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


def prepare_data(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, LoanDataPreprocessor]:
    """
    Prepares dataset for training:
    - Extracts binary target (Fully Paid = 0, Charged Off = 1)
    - Performs 80:20 Stratified train/test split per Section 3.4
    - Fits and applies LoanDataPreprocessor
    """
    from sklearn.model_selection import train_test_split

    df = df.copy()
    
    # Map target
    if df[TARGET_COL].dtype == object or isinstance(df[TARGET_COL].iloc[0], str):
        y = df[TARGET_COL].map(TARGET_MAPPING).fillna(0).astype(int)
    else:
        y = df[TARGET_COL].astype(int)
        
    X = df.drop(columns=[TARGET_COL])
    
    # 80:20 Stratified Split per paper section 3.4
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = LoanDataPreprocessor(winsorize_limits=(0.01, 0.01), scale_features=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor
