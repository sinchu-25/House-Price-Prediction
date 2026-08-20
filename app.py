import streamlit as st
import pandas as pd
import numpy as np
import joblib

from preprocessing import clean_and_engineer, encode_features, align_columns

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# ---------------------------------------------------------
# Load artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoding_maps = joblib.load("target_encoding_maps.pkl")
    encoding_fallback = joblib.load("target_encoding_fallback.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    skewed_features = joblib.load("skewed_features.pkl")
    template_row = joblib.load("template_row.pkl")
    metrics = joblib.load("metrics.pkl")
    neighborhoods = joblib.load("neighborhoods.pkl")
    return model, scaler, encoding_maps, encoding_fallback, feature_columns, skewed_features, template_row, metrics, neighborhoods

model, scaler, encoding_maps, encoding_fallback, feature_columns, skewed_features, template_row, metrics, neighborhoods = load_artifacts()

TARGET_ENC_FEATURES = ['Neighborhood', 'Exterior1st', 'Exterior2nd', 'MSSubClass', 'Condition1']

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption(
    "Gradient Boosting model trained on the Ames Housing dataset (Kaggle House Prices). "
    "Portfolio demo — adjust the key features below; everything else uses typical values "
    "from the training data."
)

with st.expander("ℹ️ About this model"):
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{metrics['r2']:.4f}")
    c2.metric("RMSE", f"${metrics['rmse']:,.0f}")
    c3.metric("MAE", f"${metrics['mae']:,.0f}")
    st.markdown(
        """
        **Pipeline:** missing-value imputation → feature engineering (house age, total
        square footage, total bathrooms, quality scores) → outlier removal → ordinal +
        one-hot + target encoding → skew correction (log1p) → feature scaling →
        Gradient Boosting Regressor (hyperparameter-tuned).

        Only the fields below are exposed for interaction — the model actually uses
        ~50+ engineered features, but most homes in the training data are similar enough
        on the rest that these are the features that move the prediction the most.
        """
    )

st.divider()

# ---------------------------------------------------------
# Inputs — simple, human-readable subset
# ---------------------------------------------------------
st.subheader("Describe the house")

col1, col2, col3 = st.columns(3)

with col1:
    overall_qual = st.slider("Overall Quality (1=Poor, 10=Excellent)", 1, 10, 6)
    gr_liv_area = st.number_input("Above-Ground Living Area (sq ft)", 300, 6000, 1500, step=50)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", 0, 3000, 800, step=50)
    lot_area = st.number_input("Lot Area (sq ft)", 1000, 50000, 9500, step=100)

with col2:
    year_built = st.slider("Year Built", 1872, 2010, 1975)
    full_bath = st.selectbox("Full Bathrooms", [0, 1, 2, 3, 4], index=2)
    bedrooms = st.selectbox("Bedrooms Above Ground", [0, 1, 2, 3, 4, 5, 6], index=3)
    garage_cars = st.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4], index=2)

with col3:
    neighborhood = st.selectbox("Neighborhood", neighborhoods, index=neighborhoods.index("NAmes") if "NAmes" in neighborhoods else 0)
    kitchen_qual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"], index=1)
    fireplaces = st.selectbox("Fireplaces", [0, 1, 2, 3], index=1)
    central_air = st.selectbox("Central Air Conditioning", ["Y", "N"], index=0)

st.divider()

# ---------------------------------------------------------
# Build the full feature row: template + overrides
# ---------------------------------------------------------
row = dict(template_row)  # typical house as the starting point
row.update({
    "OverallQual": overall_qual,
    "GrLivArea": gr_liv_area,
    "TotalBsmtSF": total_bsmt_sf,
    "LotArea": lot_area,
    "YearBuilt": year_built,
    "FullBath": full_bath,
    "BedroomAbvGr": bedrooms,
    "GarageCars": garage_cars,
    "Neighborhood": neighborhood,
    "KitchenQual": kitchen_qual,
    "Fireplaces": fireplaces,
    "CentralAir": central_air,
    "YrSold": template_row.get("YrSold", 2008),
    "YearRemodAdd": max(year_built, template_row.get("YearRemodAdd", year_built)),
    "GarageYrBlt": year_built,
})

input_df = pd.DataFrame([row])

# ---------------------------------------------------------
# Run through the SAME pipeline used in training
# ---------------------------------------------------------
def predict_price(raw_row_df):
    df = clean_and_engineer(raw_row_df)
    df = encode_features(df)

    for col in TARGET_ENC_FEATURES:
        if col in df.columns:
            val = df[col].iloc[0]
            df[col] = encoding_maps.get(col, {}).get(val, encoding_fallback.get(col, 0.0))

    df = align_columns(df, feature_columns)

    for col in skewed_features:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    scaled = scaler.transform(df)
    log_pred = model.predict(scaled)[0]
    return float(np.expm1(log_pred))

if st.button("💰 Predict Price", type="primary"):
    price = predict_price(input_df)
    st.success(f"### Estimated Sale Price: **${price:,.0f}**")
    st.caption(
        f"Model's typical error on the test set is about ±${metrics['mae']:,.0f} (MAE), "
        f"so treat this as a ballpark, not an appraisal."
    )

st.caption("Built as a portfolio project — House Price Prediction. Full training pipeline and notebook on GitHub.")