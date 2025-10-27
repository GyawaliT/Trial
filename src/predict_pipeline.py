import joblib, json, os
import numpy as np, pandas as pd
import torch
import torch.nn as nn
from config import FEATURE_LIST

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super().__init__()
        layers = []
        in_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = h
        layers.append(nn.Linear(in_size, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def load_xgb_model(models_dir="models"):
    model = joblib.load(os.path.join(models_dir, "xgb_model.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    imputer = joblib.load(os.path.join(models_dir, "imputer.joblib"))
    selector = joblib.load(os.path.join(models_dir, "selector.joblib"))
    return model, scaler, imputer, selector

def load_ann_model(models_dir="models", device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(os.path.join(models_dir, "ann_meta.json"), "r") as f:
        meta = json.load(f)
    model = NeuralNetwork(meta["input_dim"], meta["hidden_sizes"], meta["dropout"]).to(device)
    state = torch.load(os.path.join(models_dir, "ann_state.pth"), map_location=device)
    model.load_state_dict(state)
    model.eval()
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    imputer = joblib.load(os.path.join(models_dir, "imputer.joblib"))
    selector = joblib.load(os.path.join(models_dir, "selector.joblib"))
    return model, scaler, imputer, selector

def prepare_X(df_features, scaler, imputer, selector):
    X = df_features.copy()
    X = np.log1p(X)
    X_prep = selector.transform(imputer.transform(scaler.transform(X)))
    return X_prep

def predict_with_xgb(df_features, models_dir="models"):
    model, scaler, imputer, selector = load_xgb_model(models_dir)
    X_prep = prepare_X(df_features, scaler, imputer, selector)
    preds_log = model.predict(X_prep)
    return np.expm1(preds_log)

def predict_with_ann(df_features, models_dir="models"):
    model, scaler, imputer, selector = load_ann_model(models_dir)
    X_prep = prepare_X(df_features, scaler, imputer, selector)
    device = next(model.parameters()).device
    with torch.no_grad():
        xt = torch.tensor(X_prep, dtype=torch.float32).to(device)
        preds_log = model(xt).cpu().numpy().flatten()
    return np.expm1(preds_log)

if __name__ == "__main__":
    # quick demo: read a CSV of features and print predictions
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_csv", default="data/p_wave_features_dataset.csv")
    parser.add_argument("--models_dir", default="models")
    args = parser.parse_args()
    df = pd.read_csv(args.features_csv)
    df = df[FEATURE_LIST]
    print("Predicting with XGBoost...")
    print(predict_with_xgb(df, args.models_dir))
    print("Predicting with ANN...")
    print(predict_with_ann(df, args.models_dir))
