import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,               # MAE: Mean Absolute Error
    median_absolute_error,             # MedAE: Median Absolute Error
    r2_score,                          # R^2: R-squared, coefficient of determination
    max_error,                         # Max Error: Maximum Error
    make_scorer,                       # Custom scorer function
    mean_absolute_percentage_error     # MAPE: Mean Absolute Percentage Error
)

st.set_page_config(page_title="Brazilian Marketplace Price Predictor", page_icon="💸", layout="wide")
st.title("💸 Price Prediction + Optimal Listing Suggestions")

# --------------------------------------------
# CATEGORY AND STATE LISTS
# --------------------------------------------
CATEGORIES = [
    "Bed Bath Table", "Health and Beauty", "Sports", "Furniture",
    "Computer Accessories", "Housewares", "Gift", "Telephony",
    "Garden Tools", "Auto"
]

# ✅ Brazilian states (from your dataset)
STATES = [
    "MG","PR","SP","RS","RJ","GO","MA","SC","ES","DF","BA",
    "PI","RO","MT","RN","PE","CE","SE","MS","PB","PA","AM"
]

# --------------------------------------------
# PRICE BINS AND OPTIMAL VALUES
# --------------------------------------------
PRICE_BINS = {
    "Bed Bath Table": {
        "bins": [0, 48, 79, 115, 169, 2000],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
    
    "Health and Beauty": {
        "bins": [0, 40, 80, 130, 245, 3124],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },

    "Sports": {
        "bins": [0, 44, 76.5, 130, 245, 4059],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },

    "Furniture": {
        "bins": [0, 39.9, 65, 99.9, 180, 1899],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },

    "Computer Accessories": {
        "bins": [0, 50, 60, 100, 270, 3930],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
    "Housewares": {
        "bins": [0, 40, 80, 130, 382, 3124],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
    "Gift": {
        "bins": [0, 59, 129, 211.91, 460, 3999.9],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
    "Telephony": {
        "bins": [0, 21.99, 29.99, 49, 80, 2428],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
    "Garden Tools": {
        "bins": [0, 49.9, 59.9, 99.99, 270, 3930],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
    "Auto": {
        "bins": [0, 40, 85, 156, 1417, 2258],
        "labels": ["Low", "Mid", "High", "Premium", "Luxury"]
    },
}
OPTIMAL_VALUES = {
    "Bed Bath Table":{
        "Low":{"name_len":49,"desc_len":320,"photo_qty":1},
        "Mid":{"name_len":53,"desc_len":486,"photo_qty":1},
        "High":{"name_len":54,"desc_len":259,"photo_qty":1},
        "Premium":{"name_len":56,"desc_len":293,"photo_qty":1},
        "Luxury":{"name_len":56,"desc_len":289,"photo_qty":1}
    },

    "Health and Beauty":{
        "Low":{"name_len":49,"desc_len":575,"photo_qty":1},
        "Mid":{"name_len":50,"desc_len":918,"photo_qty":1},
        "High":{"name_len":52,"desc_len":1136,"photo_qty":1},
        "Premium":{"name_len":45,"desc_len":3078,"photo_qty":4}
    },

    "Sports":{
        "Low":{"name_len":47,"desc_len":618,"photo_qty":1},
        "Mid":{"name_len":49,"desc_len":805,"photo_qty":1},
        "High":{"name_len":49,"desc_len":920,"photo_qty":1},
        "Premium":{"name_len":50,"desc_len":1007,"photo_qty":1},
        "Luxury":{"name_len":51,"desc_len":1103,"photo_qty":2}
    },

    "Furniture":{
        "Low":{"name_len":49,"desc_len":546,"photo_qty":1},
        "Mid":{"name_len":55,"desc_len":599,"photo_qty":2},
        "High":{"name_len":51,"desc_len":682,"photo_qty":2},
        "Premium":{"name_len":53,"desc_len":676,"photo_qty":1}
    },

    "Computer Accessories":{
        "Low":{"name_len":47,"desc_len":671,"photo_qty":2},
        "Mid":{"name_len":49,"desc_len":464,"photo_qty":1},
        "High":{"name_len":45,"desc_len":452,"photo_qty":1},
        "Premium":{"name_len":53,"desc_len":871,"photo_qty":1},
        "Luxury":{"name_len":50,"desc_len":779,"photo_qty":1}
    },

    "Housewares":{
        "Low":{"name_len":49,"desc_len":575,"photo_qty":1},
        "Mid":{"name_len":50,"desc_len":732,"photo_qty":1},
        "High":{"name_len":50,"desc_len":919,"photo_qty":1},
        "Premium":{"name_len":52,"desc_len":1153,"photo_qty":1},
        "Luxury":{"name_len":50,"desc_len":1135,"photo_qty":2}
    },

    "Gift":{
        "Low":{"name_len":56,"desc_len":345,"photo_qty":3},
        "Mid":{"name_len":53,"desc_len":523,"photo_qty":2},
        "High":{"name_len":42,"desc_len":582,"photo_qty":2},
        "Premium":{"name_len":47,"desc_len":536,"photo_qty":1},
        "Luxury":{"name_len":47,"desc_len":589,"photo_qty":2}
    },

    "Telephony":{
        "Low":{"name_len":52,"desc_len":447,"photo_qty":1},
        "Mid":{"name_len":58,"desc_len":734,"photo_qty":1},
        "High":{"name_len":54,"desc_len":540,"photo_qty":2},
        "Premium":{"name_len":53,"desc_len":614,"photo_qty":2},
        "Luxury":{"name_len":53,"desc_len":1239,"photo_qty":2}
    },

    "Garden Tools":{
        "Low":{"name_len":57,"desc_len":348,"photo_qty":2},
        "Mid":{"name_len":56,"desc_len":399,"photo_qty":2},
        "High":{"name_len":39,"desc_len":1654,"photo_qty":2},
        "Premium":{"name_len":46,"desc_len":825,"photo_qty":1},
        "Luxury":{"name_len":55,"desc_len":1125,"photo_qty":2}
    },

    "Auto":{
        "Low":{"name_len":55,"desc_len":589,"photo_qty":2},
        "Mid":{"name_len":56,"desc_len":695,"photo_qty":2},
        "High":{"name_len":54,"desc_len":685,"photo_qty":2},
        "Premium":{"name_len":54,"desc_len":829,"photo_qty":2},
        "Luxury":{"name_len":56,"desc_len":2106,"photo_qty":1}
    }
}

# --------------------------------------------
# MODEL PIPELINE (with your features)
# --------------------------------------------
FEATURES_NUM = [
    "product_description_lenght",
    "product_photos_qty",
    "product_weight_g",
    "product_length_cm",
    "product_height_cm",
    "product_width_cm",
]
FEATURES_CAT = ["seller_state", "product_category_name_english"]

def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("num", RobustScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    return Pipeline([("pre", pre), ("model", model)])

# --------------------------------------------
# Create demo synthetic model
# --------------------------------------------
def synth_model() -> Pipeline:
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(1200):
        desc = rng.integers(100, 1000)
        photos = rng.integers(1, 8)
        w = rng.normal(500, 150)
        L = rng.normal(20, 5)
        H = rng.normal(10, 3)
        W = rng.normal(12, 4)
        state = rng.choice(STATES)
        cat = rng.choice(CATEGORIES)
        price = (0.02*w + 0.5*photos + 0.1*desc + 0.3*L + 0.2*H + rng.normal(0, 20))
        rows.append([desc, photos, w, L, H, W, state, cat, max(10, float(price))])
    df = pd.DataFrame(rows, columns=FEATURES_NUM + FEATURES_CAT + ["price"])
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df["price"]
    pipe = build_model()
    pipe.fit(X, y)
    return pipe

def format_brl(value: float) -> str:
    s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def assign_price_bin(category, price):
    spec = PRICE_BINS.get(category)
    if not spec: return None
    result = pd.cut([price], bins=spec["bins"], labels=spec["labels"], include_lowest=True)
    return str(result[0]) if len(result) else None

# --------------------------------------------
# STREAMLIT UI
# --------------------------------------------
st.subheader("1️⃣ Select Product Category")
category = st.selectbox("product_category_name_english", options=CATEGORIES, index=0)

st.subheader("2️⃣ Enter Product Features")
c1, c2, c3 = st.columns(3)
with c1:
    product_description_lenght = st.number_input("product_description_lenght", min_value=10, value=300, step=1)
    product_photos_qty = st.number_input("product_photos_qty", min_value=1, value=3, step=1)
    product_weight_g = st.number_input("product_weight_g", min_value=0.0, value=500.0, step=1.0)
with c2:
    product_length_cm = st.number_input("product_length_cm", min_value=0.0, value=20.0, step=0.5)
    product_height_cm = st.number_input("product_height_cm", min_value=0.0, value=10.0, step=0.5)
    product_width_cm = st.number_input("product_width_cm", min_value=0.0, value=12.0, step=0.5)
with c3:
    seller_state = st.selectbox("seller_state", options=STATES)

X_input = pd.DataFrame([{
    "product_description_lenght": product_description_lenght,
    "product_photos_qty": product_photos_qty,
    "product_weight_g": product_weight_g,
    "product_length_cm": product_length_cm,
    "product_height_cm": product_height_cm,
    "product_width_cm": product_width_cm,
    "seller_state": seller_state,
    "product_category_name_english": category,
}])

# --------------------------------------------
# Run model prediction
# --------------------------------------------
if "pipe" not in st.session_state:
    st.session_state["pipe"] = synth_model()
pipe = st.session_state["pipe"]

st.subheader("3️⃣ Predicted Price")
pred_price = float(pipe.predict(X_input)[0])
st.metric("Predicted Price", format_brl(pred_price))


# --------------------------------------------
# Suggest optimal listing settings
# --------------------------------------------
st.subheader("4️⃣ Suggested Optimal Listing Settings")
price_bin = assign_price_bin(category, pred_price)

if price_bin is None or price_bin == "nan":
    st.warning("Price is outside bin range for this category.")
else:
    opt = OPTIMAL_VALUES.get(category, {}).get(price_bin)
    if opt:
        c1, c2, c3 = st.columns(3)
        c1.metric("📝 Optimal Name Length", f"{opt['name_len']} chars")
        c2.metric("📄 Optimal Description Length", f"{opt['desc_len']} chars")
        c3.metric("📷 Optimal Photo Qty", opt["photo_qty"])
    else:
        st.info(f"No suggestion data available for '{category}' at tier '{price_bin}'.")

st.caption("This model uses product features, category, and seller_state to predict the price and recommend optimal listing details.")