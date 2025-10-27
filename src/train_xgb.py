import os, argparse
import numpy as np, pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocessing import fit_preprocessing
from config import FEATURE_LIST

def evaluate_df(y_true_log, y_pred_log, y_true_raw, y_pred_raw):
    return {
        'R2_log': r2_score(y_true_log, y_pred_log),
        'MAE_log': mean_absolute_error(y_true_log, y_pred_log),
        'RMSE_log': float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        'R2_raw': r2_score(y_true_raw, y_pred_raw),
        'MAE_raw': mean_absolute_error(y_true_raw, y_pred_raw),
        'RMSE_raw': float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw))),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/EEW_features_2024-10-21.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data, skiprows=[1])
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if c not in ['filename','date','time']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))

    df = df[(df[FEATURE_LIST] > 0).all(axis=1)]
    X = df[FEATURE_LIST]
    y_raw = df['PGA']
    X_log = np.log1p(X)
    y_log = np.log1p(y_raw)

    # stratified split (train/val/test)
    y_bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(sss1.split(X_log, y_bins))
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X_log.iloc[temp_idx], y_bins.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

   # keep columns names so transformers that depend on feature names won't warn
    X_train_pre = imputer.transform(scaler.transform(np.log1p(X_train)))
    X_train_pre = pd.DataFrame(X_train_pre, columns=X_train.columns, index=X_train.index)
    X_train_sel = selector.transform(X_train_pre)

    X_val_pre = imputer.transform(scaler.transform(np.log1p(X_val)))
    X_val_pre = pd.DataFrame(X_val_pre, columns=X_val.columns, index=X_val.index)
    X_val_sel = selector.transform(X_val_pre)

    X_test_pre = imputer.transform(scaler.transform(np.log1p(X_test)))
    X_test_pre = pd.DataFrame(X_test_pre, columns=X_test.columns, index=X_test.index)
    X_test_sel = selector.transform(X_test_pre)

    scaler, imputer, selector = fit_preprocessing(X_train, y_train_log, args.out)

    # use the best params you found previously (from Codes.py)
    best_params = {'n_estimators': 776, 'learning_rate': 0.010590433420511285,
                   'max_depth': 6, 'subsample': 0.666852461341688,
                   'colsample_bytree': 0.8724127328229327}
    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=best_params['n_estimators'],
                             learning_rate=best_params['learning_rate'],
                             max_depth=best_params['max_depth'],
                             subsample=best_params['subsample'],
                             colsample_bytree=best_params['colsample_bytree'],
                             random_state=42)
    model.fit(X_train_sel, y_train_log)

    # predictions
    pred_test_log = model.predict(X_test_sel)
    pred_test_raw = np.expm1(pred_test_log)

    metrics = evaluate_df(y_test_log, pred_test_log, y_test_raw, pred_test_raw)
    print("Test metrics:", metrics)

    joblib.dump(model, f"{args.out}/xgb_model.joblib")
    print(f"Saved XGBoost model to {args.out}/xgb_model.joblib")
