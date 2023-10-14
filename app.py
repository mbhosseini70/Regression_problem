import streamlit as st
import catboost
import numpy as np
import pandas as pd
import joblib

from functions import preprocess_data, sort_addresses_with_other_at_end
from functions import columns, addresses


st.title("House Information App")
st.title ("Welcome to our platform!")

st.write ("Just share your desired house features with me, and I'll provide you with an estimated price in Rials.")




default_value = 100.0
formatted_default = float(f"{default_value:.1f}")
area = st.number_input("Enter the area of the house:", min_value=0.0, step=0.1, value=formatted_default)

room = st.selectbox("Select number of rooms:", [i for i in range(6)],2)

parking = st.radio("Parking:", ["Has Parking", "Does not have Parking"])
parking = 1 if parking == "Has Parking" else 0


warehouse = st.radio("Warehouse:", ["Has Warehouse", "Does not have Warehouse"])
warehouse = 1 if warehouse == "Has Warehouse" else 0


elevator = st.radio("Elevator:", ["Has Elevator", "Does not have Elevator"])
elevator = 1 if elevator == "Has Elevator" else 0

address = st.selectbox("Select Address:", sort_addresses_with_other_at_end(addresses), help="Start typing to filter")





def predict_with_cat():
    cat_model = catboost.CatBoost()
    cat_model.load_model('CatBoost_model.cbm')
    row = np.array([area, room, parking, warehouse, elevator, address])
    X = pd.DataFrame([row], columns = columns )
    X = preprocess_data(X)
    result = cat_model.predict(X)
    formatted_result = '{:,.0f}'.format(result[0])
    st.write(formatted_result)


def predict_with_xgb():
    xgb_model = joblib.load('xgboost_model.pkl')
    row = np.array([area, room, parking, warehouse, elevator, address])
    X = pd.DataFrame([row], columns = columns )
    X = preprocess_data(X)
    result = xgb_model.predict(X)
    formatted_result = '{:,.0f}'.format(result[0])
    st.write(formatted_result)    

trigger_1 = st.button("Predict with first model")
if trigger_1:
    predict_with_cat()


trigger_2 = st.button("Predict with second model")
if trigger_2:
    predict_with_xgb()