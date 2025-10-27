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

    # --- Load dataset ---
    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        if c not in ['filename','date','time']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))

    # --- Filter valid P-wave features ---
    from config import FEATURE_LIST
    df = df[(df[FEATURE_LIST] > 0).all(axis=1)]
    X = df[FEATURE_LIST]
    y_raw = df['PGA']
    X_log = np.log1p(X)
    y_log = np.log1p(y_raw)

    # --- Split into train, val, test ---
    y_bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(sss1.split(X_log, y_bins))
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X_log.iloc[temp_idx], y_bins.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train_log, y_val_log, y_test_log = y_log.iloc[train_idx], y_log.iloc[val_idx], y_log.iloc[test_idx]
    y_train_raw, y_val_raw, y_test_raw = y_raw.iloc[train_idx], y_raw.iloc[val_idx], y_raw.iloc[test_idx]

    # --- Fit preprocessing objects ---
    scaler, imputer, selector = fit_preprocessing(X_train, y_train_log, args.out)

    # --- Apply preprocessing pipeline safely (retain names) ---
    def prep(df_in):
        Xp = imputer.transform(scaler.transform(np.log1p(df_in)))
        Xp = pd.DataFrame(Xp, columns=df_in.columns, index=df_in.index)
        return selector.transform(Xp)

    X_train_sel = prep(X_train)
    X_val_sel = prep(X_val)
    X_test_sel = prep(X_test)

    # --- Train XGBoost model ---
    best_params = {'n_estimators': 776, 'learning_rate': 0.010590433420511285,
                   'max_depth': 6, 'subsample': 0.666852461341688,
                   'colsample_bytree': 0.8724127328229327}

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42
    )

    model.fit(X_train_sel, y_train_log)

    # --- Evaluate model ---
    pred_test_log = model.predict(X_test_sel)
    pred_test_raw = np.expm1(pred_test_log)
    metrics = evaluate_df(y_test_log, pred_test_log, y_test_raw, pred_test_raw)

    print("\n✅ XGBoost Training Complete")
    print("📊 Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Save model and preprocessing ---
    joblib.dump(model, f"{args.out}/xgb_model.joblib")
    print(f"\n💾 Saved model to {args.out}/xgb_model.joblib")
