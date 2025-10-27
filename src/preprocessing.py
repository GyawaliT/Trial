import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

def fit_preprocessing(X_df, y_log, save_dir):
    """Fit scaler/imputer/selector and save them."""
    scaler = RobustScaler().fit(X_df)
    imputer = SimpleImputer(strategy='mean').fit(X_df)
    selector = SelectKBest(score_func=f_regression, k='all').fit(X_df, y_log)

    joblib.dump(scaler, f"{save_dir}/scaler.joblib")
    joblib.dump(imputer, f"{save_dir}/imputer.joblib")
    joblib.dump(selector, f"{save_dir}/selector.joblib")
    return scaler, imputer, selector

def load_preprocessing(load_dir):
    scaler = joblib.load(f"{load_dir}/scaler.joblib")
    imputer = joblib.load(f"{load_dir}/imputer.joblib")
    selector = joblib.load(f"{load_dir}/selector.joblib")
    return scaler, imputer, selector

def transform_features(df_features, scaler, imputer, selector):
    """Apply scaler, imputer, selector; expects pandas DataFrame of feature columns."""
    X = df_features.copy()
    # apply log1p to input features (as in training)
    X = np.log1p(X)
    X_prep = selector.transform(imputer.transform(scaler.transform(X)))
    return X_prep
