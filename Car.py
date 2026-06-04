import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)
st.markdown("""
<style>

/* Car Background */
.stApp{
    background:
    linear-gradient(
        rgba(0,0,0,0.25),
        rgba(0,0,0,0.25)
    ),
    url("https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=1600&q=80");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* All Text Visible */
h1, h2, h3, h4, h5, h6,
p, label, span, div {
    color: white !important;
}

/* Glass Card */
.glass-card{
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    padding:25px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,0.30);
}

/* Main Title */
.main-title{
    text-align:center;
    color:white;
    font-size:50px;
    font-weight:700;
}

/* Subtitle */
.sub-title{
    text-align:center;
    color:white;
    font-size:20px;
    margin-bottom:25px;
}

/* Input Labels */
label{
    color:white !important;
    font-size:16px !important;
    font-weight:bold !important;
}

/* Buttons */
.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:linear-gradient(
        90deg,
        #22c55e,
        #16a34a
    );
    color:white;
    font-size:20px;
    font-weight:bold;
}

/* Prediction Card */
.prediction-card{
    background:linear-gradient(
        90deg,
        #10b981,
        #059669
    );
    padding:30px;
    border-radius:18px;
    text-align:center;
    color:white;
    font-size:34px;
    font-weight:bold;
    margin-top:20px;
}
/* Footer */
.footer{
    text-align:center;
    color:white;
    margin-top:40px;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)
# -------------------------
# HEADER
# -------------------------

st.markdown("""
<div class='main-title'>
🚗 Car Price Prediction Dashboard
</div>

<div class='sub-title'>
AI Powered Used Car Price Estimation System
</div>
""", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------

df = pd.read_csv("carprice.csv")



df.replace("?", pd.NA, inplace=True)
df.dropna(inplace=True)

# -------------------------
# MODEL TRAINING
# -------------------------

y = df["price"]
X = df.drop("price", axis=1)

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------
# USER INPUT
# -------------------------

st.markdown("""
<div class='glass-card'>
<h3 style='color:white;text-align:center;'>
🚘 Enter Car Details
</h3>
</div>
""", unsafe_allow_html=True)

horsepower = st.number_input(
    "Horse Power",
    min_value=40,
    max_value=500,
    value=100
)

enginesize = st.number_input(
    "Engine Size",
    min_value=50,
    max_value=500,
    value=120
)

citympg = st.number_input(
    "City MPG",
    min_value=5,
    max_value=60,
    value=25
)

highwaympg = st.number_input(
    "Highway MPG",
    min_value=5,
    max_value=70,
    value=30
)

# -------------------------
# SHOW CAR DETAILS
# -------------------------

with st.expander("🚘 Show Car Details"):

    st.subheader("Car Dataset Details")
    st.dataframe(df.head(20))

    # -------------------------
# AVAILABLE FEATURES
# -------------------------

with st.expander("📋 Available Features"):

    feature_df = pd.DataFrame({
        "Feature Name": X.columns
    })

    st.dataframe(
        feature_df,
        use_container_width=True
    )    
# -------------------------
# PREDICTION
# -------------------------

if st.button("Predict Car Price"):

    sample = X.iloc[[0]].copy()

    if "horsepower" in sample.columns:
        sample["horsepower"] = horsepower

    if "enginesize" in sample.columns:
        sample["enginesize"] = enginesize

    if "citympg" in sample.columns:
        sample["citympg"] = citympg

    if "highwaympg" in sample.columns:
        sample["highwaympg"] = highwaympg

    predicted_price = model.predict(sample)[0]

    st.markdown(
        f"""
        <div class='prediction-card'>
            Estimated Car Price <br>
            ₹ {predicted_price:,.0f}
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# FOOTER
# -------------------------

st.markdown("""
<div class='footer'>
</div>
""", unsafe_allow_html=True)