import os, json, argparse
import numpy as np, pandas as pd
import joblib
import torch, torch.nn as nn, torch.optim as optim
from sklearn.model_selection import StratifiedShuffleSplit
from preprocessing import fit_preprocessing
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

def train_model(X_train, y_train, X_val, y_val, hidden_sizes=[428,442,220], lr=0.0011676, dropout=0.28358, weight_decay=6.37045e-05, epochs=829, device='cpu'):
    device = torch.device(device)
    model = NeuralNetwork(X_train.shape[1], hidden_sizes, dropout).to(device)
    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1).to(device)
    if X_val is not None:
        X_va = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_va = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1).to(device)

    best_loss = float('inf')
    patience = 25; wait = 0
    best_state = None

    for epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr), y_tr)
        loss.backward()
        optimizer.step()

        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_va), y_va).item()
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

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

    y_bins = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(sss1.split(X_log, y_bins))
    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X_log.iloc[temp_idx], y_bins.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    X_train_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx]
    y_train_log, y_val_log = y_log.iloc[train_idx], y_log.iloc[val_idx]

    scaler, imputer, selector = fit_preprocessing(X_train_df, y_train_log, args.out)
    X_train_sel = selector.transform(imputer.transform(scaler.transform(np.log1p(X_train_df))))
    X_val_sel = selector.transform(imputer.transform(scaler.transform(np.log1p(X_val_df))))

    # best hyperparams from your notebook
    hidden_sizes = [428,442,220]
    lr = 0.0011676487575205433
    dropout = 0.2835776221997114
    weight_decay = 6.370451204388144e-05
    epochs = 829

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_model(X_train_sel, y_train_log, X_val_sel, y_val_log,
                        hidden_sizes=hidden_sizes, lr=lr, dropout=dropout,
                        weight_decay=weight_decay, epochs=epochs, device=device)

    # save
    torch.save(model.state_dict(), f"{args.out}/ann_state.pth")
    meta = {
        "input_dim": X_train_sel.shape[1],
        "hidden_sizes": hidden_sizes,
        "lr": lr,
        "dropout": dropout,
        "weight_decay": weight_decay,
        "epochs": epochs
    }
    with open(f"{args.out}/ann_meta.json", "w") as f:
        json.dump(meta, f)
    print(f"Saved ANN state and meta to {args.out}")
