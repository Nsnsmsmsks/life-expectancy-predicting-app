import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# App title
st.title("Life Expectancy Prediction App")
st.write("This app predicts life expectancy using a simple machine learning model.")

# Sample dataset (you can replace with your real data later)
data = {
    "GDP": [1000, 2000, 3000, 4000, 5000],
    "Schooling": [8, 9, 10, 11, 12],
    "LifeExpectancy": [55, 60, 65, 70, 75]
}

df = pd.DataFrame(data)

# Train model
X = df[["GDP", "Schooling"]]
y = df["LifeExpectancy"]

model = LinearRegression()
model.fit(X, y)

# User inputs
st.subheader("Enter input values")
gdp = st.number_input("GDP per capita", min_value=0.0, value=2000.0)
schooling = st.number_input("Average years of schooling", min_value=0.0, value=10.0)

# Prediction
if st.button("Predict Life Expectancy"):
    prediction = model.predict([[gdp, schooling]])
    st.success(f"Predicted Life Expectancy: {prediction[0]:.2f} years")

