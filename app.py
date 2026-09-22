import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv("house_data.csv")

X = df[["size", "bedrooms", "bathrooms", "age", "parking"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠"
)
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #1f3c88;
    text-align: center;
}
.stButton > button {
    width: 100%;
    background-color: #1f3c88;
    color: white;
    border-radius: 10px;
    padding: 12px;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)
st.title("🏠 House Price Prediction")
st.markdown(
    "### Smart ML-based property price estimation"
)
st.caption("Enter the house details to get an estimated market price.")
st.divider()
col1, col2 = st.columns(2)

with col1:
    size = st.number_input("House Size (sq.ft)", 500, 5000, 1500)
    bedrooms = st.number_input("Bedrooms", 1, 10, 3)
    bathrooms = st.number_input("Bathrooms", 1, 10, 2)

with col2:
    age = st.number_input("House Age (years)", 0, 50, 5)
    parking = st.number_input("Parking Spaces", 0, 5, 1)

if st.button("Predict Price"):
    data = pd.DataFrame({
        "size": [size],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "age": [age],
        "parking": [parking]
    })

    price = model.predict(data)[0]

    st.success(f"Predicted Price: ₹{price:,.2f}")