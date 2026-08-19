# ============================================================
# House Price Prediction — retrain for deployment
# Works in Jupyter Notebook / JupyterLab (or Colab) — needs internet for pip installs
# ============================================================
#
# STEP 1: Put house_price_pred.csv AND preprocessing.py in the SAME FOLDER as this script
# STEP 2: Run this whole script as one cell (or paste section by section)
# STEP 3: Everything gets saved into a ./deploy_artifacts folder, plus a zip

get_ipython().system('pip install -q category_encoders lightgbm')

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from category_encoders import TargetEncoder

from preprocessing import clean_and_engineer, encode_features, align_columns, TARGET_ENC_FEATURES

SAVE_DIR = "./deploy_artifacts"
os.makedirs(SAVE_DIR, exist_ok=True)
RANDOM_STATE = 42
SEARCH_ITERS = 20  # fewer = faster, still finds a solid model

# ---- 1. Load data ----
df = pd.read_csv("house_price_pred.csv")
print("Loaded:", df.shape)

# ---- 2. Clean + engineer (shared logic with the app) ----
df = clean_and_engineer(df)
print("After cleaning + engineering:", df.shape)

# ---- 3. Outlier removal (same rules as your notebook) ----
before = df.shape[0]
df = df.drop(df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 200000)].index)
df = df.drop(df[df['LotArea'] > 100000].index)
print(f"Outliers removed: {before - df.shape[0]}")

# ---- 4. Low-correlation feature drop (numeric features, corr with SalePrice < 0.05) ----
num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols.remove('SalePrice')
corr_with_target = df[num_cols].corrwith(df['SalePrice']).abs()
low_corr = corr_with_target[corr_with_target < 0.05].index.tolist()
df = df.drop(columns=low_corr)
print(f"Dropped {len(low_corr)} low-correlation features:", low_corr)

# ---- 5. Multicollinearity drop (numeric pairs with corr > 0.85, keep the one more correlated with target) ----
num_cols = df.select_dtypes(include='number').columns.tolist()
if 'SalePrice' in num_cols:
    num_cols.remove('SalePrice')
corr_matrix = df[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_multicollinear = set()
for col in upper.columns:
    for row in upper.index:
        val = upper.loc[row, col]
        if pd.notna(val) and val > 0.85:
            corr_row = abs(df[row].corr(df['SalePrice']))
            corr_col = abs(df[col].corr(df['SalePrice']))
            drop_multicollinear.add(row if corr_row < corr_col else col)
df = df.drop(columns=list(drop_multicollinear))
print(f"Dropped {len(drop_multicollinear)} multicollinear features:", drop_multicollinear)

# ---- 6. Ordinal + one-hot encoding (shared logic) ----
df = encode_features(df)
print("After ordinal + one-hot encoding:", df.shape)

# ---- 7. Skew treatment on numeric features (excluding SalePrice, computed before target encoding) ----
num_cols = df.select_dtypes(include='number').columns.tolist()
num_cols = [c for c in num_cols if c != 'SalePrice' and c not in TARGET_ENC_FEATURES]
skewness = df[num_cols].skew()
skewed_features = skewness[abs(skewness) > 0.5].index.tolist()
for col in skewed_features:
    df[col] = np.log1p(df[col].clip(lower=0))
print(f"Log1p applied to {len(skewed_features)} skewed features")

# ---- 8. Log-transform target ----
df['SalePrice'] = np.log1p(df['SalePrice'])

# ---- 9. Train/test split BEFORE target encoding + scaling (avoid leakage) ----
X = df.drop(columns=['SalePrice'])
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

# ---- 10. Target encoding — fit on TRAIN ONLY ----
target_enc_present = [c for c in TARGET_ENC_FEATURES if c in X_train.columns]
encoder = TargetEncoder(smoothing=10)
X_train[target_enc_present] = encoder.fit_transform(X_train[target_enc_present], y_train)
X_test[target_enc_present] = encoder.transform(X_test[target_enc_present])

feature_columns = X_train.columns.tolist()

# ---- 11. Scaling — fit on TRAIN ONLY ----
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_columns, index=X_test.index)

# ---- 12. Train Gradient Boosting (lighter random search than the original notebook, same model family) ----
param_dist = {
    'n_estimators': [500, 700, 800, 900, 1000],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.7, 0.8, 0.9, 1.0],
}
search = RandomizedSearchCV(
    estimator=GradientBoostingRegressor(random_state=RANDOM_STATE),
    param_distributions=param_dist,
    n_iter=SEARCH_ITERS,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=RANDOM_STATE,
    verbose=1,
)
search.fit(X_train_scaled, y_train)
gb_model = search.best_estimator_
print("Best params:", search.best_params_)

# ---- 13. Evaluate (reverse log transform to real dollar terms) ----
preds = gb_model.predict(X_test_scaled)
y_test_actual = np.expm1(y_test)
preds_actual = np.expm1(preds)

metrics = {
    'r2': r2_score(y_test_actual, preds_actual),
    'rmse': float(np.sqrt(mean_squared_error(y_test_actual, preds_actual))),
    'mae': float(mean_absolute_error(y_test_actual, preds_actual)),
}
print("Final metrics:", metrics)

# ---- 14. Build a "typical house" template for the app's default values ----
# IMPORTANT: built from RAW columns (before clean_and_engineer), because the app
# re-runs clean_and_engineer() on its input row from scratch — the template must
# supply every raw column that step expects (GarageQual, YearBuilt, PoolArea, etc.),
# not the already-engineered/dropped columns.
df_raw_source = pd.read_csv("house_price_pred.csv")
template_row = {}
for col in df_raw_source.columns:
    if col in ('SalePrice', 'Id'):
        continue
    if pd.api.types.is_numeric_dtype(df_raw_source[col]):
        val = df_raw_source[col].median()
        template_row[col] = float(val) if pd.notna(val) else 0.0
    else:
        mode_vals = df_raw_source[col].mode()
        template_row[col] = mode_vals[0] if len(mode_vals) > 0 else 'None'

# ---- 15. Save everything the app needs ----
joblib.dump(gb_model, f"{SAVE_DIR}/model.pkl")
joblib.dump(scaler, f"{SAVE_DIR}/scaler.pkl")
joblib.dump(encoder, f"{SAVE_DIR}/target_encoder.pkl")
joblib.dump(feature_columns, f"{SAVE_DIR}/feature_columns.pkl")
joblib.dump(skewed_features, f"{SAVE_DIR}/skewed_features.pkl")
joblib.dump(template_row, f"{SAVE_DIR}/template_row.pkl")
joblib.dump(metrics, f"{SAVE_DIR}/metrics.pkl")
joblib.dump(sorted(df_raw_source['Neighborhood'].dropna().unique().tolist()), f"{SAVE_DIR}/neighborhoods.pkl")

print("\nSaved all artifacts to", SAVE_DIR)
print(os.listdir(SAVE_DIR))

import shutil
shutil.make_archive("deploy_artifacts", "zip", SAVE_DIR)
print("\ndeploy_artifacts.zip created in the same folder as this script.")