import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# random seeds
rng = 42
np.random.seed(rng)
torch.manual_seed(rng)

# use MPS (from m4 pro macbook pro)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print("device ->", device)

# -----------------------------
# load csv and clean up data
# -----------------------------
csv_path = "data/listings.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError("listings csv data not found")

print("reading csv...")
df = pd.read_csv(csv_path, low_memory=False)

cols_we_keep = [
    "price", "neighbourhood_cleansed", "room_type",
    "accommodates", "bedrooms", "bathrooms",
    "minimum_nights", "availability_365"
]
df = df[cols_we_keep].copy()  # copying for

# step 2: clean stuff like "$1,234.00" -> 1234.0
df["price"] = (
    df["price"].astype(str)
               .str.replace(r"[\$,]", "", regex=True)
               .astype(float)
)

# bathrooms can be strings like "1.5 baths" 
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")

# drop rows where important things are missing
df = df.dropna(subset=[
    "price", "accommodates", "bedrooms", "bathrooms",
    "minimum_nights", "availability_365",
    "neighbourhood_cleansed", "room_type"
])

# remove outliers
df = df[df["price"].between(20, 1200)]

# target: log1p helps with skew
y = np.log1p(df["price"].values.astype(np.float32))

# features split (numerical vs category ish)
num_feats   = ["accommodates", "bedrooms", "bathrooms", "minimum_nights", "availability_365"]
cat_feats   = ["neighbourhood_cleansed", "room_type"]

X_num = df[num_feats]
X_cat = df[cat_feats]

X_all = pd.concat([X_num, X_cat], axis=1)

# -----------------------------
# train/val/test split (70/15/15 )
# -----------------------------
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_all, y, test_size=0.15, random_state=rng
)

X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=rng
)  # (0.85 * 0.1765 ≈ 0.15)

print("shapes:",
      "\n  train ->", X_train.shape,
      "\n  val   ->", X_val.shape,
      "\n  test  ->", X_test.shape)

# preprocessing (ohe + standardize)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# make OHE dense
ohe = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.2

pre = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_feats),
        ("cat", ohe,              cat_feats),
    ]
)

print("fitting preprocessor...")
X_tr = pre.fit_transform(X_train)
X_va = pre.transform(X_val)
X_te = pre.transform(X_test)

cat_names = pre.named_transformers_["cat"].get_feature_names_out(cat_feats)
feat_names = np.concatenate([np.array(num_feats), cat_names])

X_tr = X_tr.astype(np.float32)
X_va = X_va.astype(np.float32)
X_te = X_te.astype(np.float32)

# -----------------------------
# torch datasets
# -----------------------------
def mk_loader(X, y_vec, bs, shuf):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y_vec, dtype=torch.float32))
    return DataLoader(ds, batch_size=bs, shuffle=shuf)

train_dl = mk_loader(X_tr, y_train, 256, True)
val_dl   = mk_loader(X_va, y_val,   4096, False)
test_dl  = mk_loader(X_te, y_test,  4096, False)

# -----------------------------
# the model
# -----------------------------
class MyLittleLinear(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.lin = nn.Linear(n, 1, bias=True) 
    def forward(self, x):
        return self.lin(x).squeeze(-1)  # squish to 1D

model = MyLittleLinear(X_tr.shape[1]).to(device)


opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
crit = nn.MSELoss()

# -----------------------------
#  training loop w/ early stopping (simple version)
# -----------------------------
EPOCHS = 500
PATIENCE = 20
best_val = float("inf")
best_state = None
wait = 0
hist_tr = []
hist_va = []

print("training...")
for ep in range(EPOCHS):
    model.train()
    tsum = 0.0; n = 0
    for xb, yb in train_dl:
        xb = xb.to(device); yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)
        lss = crit(out, yb)
        lss.backward()
        opt.step()
        tsum += lss.item() * len(xb); n += len(xb)
    tr_loss = tsum / n

    model.eval()
    with torch.no_grad():
        vsum = 0.0; m = 0
        for xb, yb in val_dl:
            xb = xb.to(device); yb = yb.to(device)
            vv = model(xb)
            ll = crit(vv, yb)
            vsum += ll.item() * len(xb); m += len(xb)
        va_loss = vsum / m

    hist_tr.append(tr_loss); hist_va.append(va_loss)

    # early stop logic
    if va_loss < best_val - 1e-7:
        best_val = va_loss
        best_state = model.state_dict()
        wait = 0
    else:
        wait += 1
        if wait >= PATIENCE:
            print("stopping early @ epoch", ep)
            break

# load the "best" weights
if best_state is not None:
    model.load_state_dict(best_state)

# -----------------------------
# test eval (undo the log)
# -----------------------------
def unlog(v):
    return np.expm1(v)

model.eval()
with torch.no_grad():
    preds_log = []
    for xb, _ in test_dl:
        xb = xb.to(device)
        preds_log.append(model(xb).cpu().numpy())
    preds_log = np.concatenate(preds_log)

y_true = unlog(y_test)
y_pred = unlog(preds_log)

MAE  = float(np.mean(np.abs(y_true - y_pred)))
RMSE = float(np.sqrt(np.mean((y_true - y_pred)**2)))
R2   = float(1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))

print("results (on $ scale) ->",
      {"MAE($)": round(MAE, 2), "RMSE($)": round(RMSE, 2), "R2": round(R2, 4)})

# -----------------------------
# plots
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

# loss curve
plt.figure(figsize=(6,4))
plt.plot(hist_tr, label="train")
plt.plot(hist_va, label="val")
plt.xlabel("epoch"); plt.ylabel("mse (log target)")
plt.title("Loss Curve"); plt.legend()
plt.tight_layout()
plt.savefig("artifacts/loss_curve.png", dpi=222)

# predicted vs actual (classic scatter with y=x line)
plt.figure(figsize=(5,5))
plt.scatter(y_true, y_pred, s=6, alpha=0.6)
a, b = y_true.min(), y_true.max()
plt.plot([a, b], [a, b])
plt.xlabel("Actual price ($)")
plt.ylabel("Predicted price ($)")
plt.title("Predicted vs Actual (Test)")
plt.tight_layout()
plt.savefig("artifacts/pred_vs_actual.png", dpi=222)

# bar chart of biggest weights (standardized)
w = model.lin.weight.detach().cpu().numpy().ravel()
idx = np.argsort(-np.abs(w))[:20]   # top 20
labels = [
    feat_names[i].replace("neighbourhood_cleansed_", "")
                 .replace("room_type_", "RT: ")
    for i in idx
]

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(labels[::-1], w[idx][::-1])
ax.set_title("Top Feature Weights")
ax.set_xlabel("weight")
fig.tight_layout()
fig.subplots_adjust(left=0.35)     # extra space for long names
plt.savefig("artifacts/top_weights.png", dpi=212, bbox_inches="tight")


# saving for future
joblib.dump(pre, "artifacts/preprocessor.joblib")
torch.save(model.state_dict(), "artifacts/model.pt")

meta = {
    "device": device,
    "n_train": int(X_tr.shape[0]),
    "n_features": int(X_tr.shape[1]),
    "metrics": {"MAE_$": MAE, "RMSE_$": RMSE, "R2": R2},
    "note": "Target was log1p(price); metrics are on original $ scale."
}
json.dump(meta, open("artifacts/metadata.json", "w"), indent=2)

print("\nall done!")
