This is my work with Torch and scikit to build a linear regression model that predicts Toronto Airbnb nightly prices.

<p align="center"> <img src="artifacts/pred_vs_actual.png" width="420"/> <img src="artifacts/loss_curve.png" width="420"/> <img src="artifacts/top_weights.png" width="420"/></p>

Predicted vs Actual - Each point is a listing. The closer to the diagonal, the better the prediction.

Loss Curve - Training and validation MSE over epochs; early stopping kicks in when validation no longer improves.

Top Feature Weights — Largest positive/negative weights in the standardized / one-hot space.

**How it works:**
  1) Loads & clean csv data.  
  2) Split into train/val/test (~70/15/15).  
  3) Preprocess with **scikit-learn**: StandardScaler for numeric features and OneHotEncoder (drop-first) for categorical features.  
  4) Train a **PyTorch** linear regressor (Adam + L2, early stopping on validation loss).  
  5) Evaluate on the test set, undoing the log transform to report MAE / RMSE / R^2 on the original $ scale.  
  6) Save artifacts.

---

**Data**
- Source: Listing data is sourced from https://insideairbnb.com/get-the-data/ (Toronto)
- Target: nightly `price` (USD), parsed from strings like `$123.00` -> `123.0`  
- Features:
  - Numeric: `accommodates`, `bedrooms`, `bathrooms`, `minimum_nights`, `availability_365`
  - Categorical (one-hot): `neighbourhood_cleansed`, `room_type`
- Basic cleaning: drop NAs in the above columns, keep prices in a reasonable range
