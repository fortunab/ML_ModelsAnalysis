import time
from math import sqrt

import numpy
import yaml
from sklearn import tree
from sklearn.linear_model import LinearRegression

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from streamlit_authenticator import Authenticate
from xgboost import XGBClassifier
from footerul import footer
from Dataset_processing import matricea_heatmap, matricea_heatmap_var_ind, modelarea
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


st.set_page_config(
        page_title="ML Methods Analysis: Regression models",
        page_icon="🎫"
    )

st.markdown("<h1> Regression Model considering both DS1 and DS2 datasets </h1>", unsafe_allow_html=True)

def modelul():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_) # fitting, potrivirea de date
    return model

def predictie_concret(real):
    model = modelul()
    y_tara_tc = model.predict(real)
    return y_tara_tc

st.header("Manual input for concrete data, Total Cases prediction ")
def predictie_users():
    with st.form("predictie"):
            teritoriul = st.text_input("Introduce Country/Others name: ", "Default")
            a = st.number_input("Introduce Population: ", 1, 100000000000, 40000, 1)
            b = st.number_input("Introduce Total Tests: ", 1, 100000000000, 500000, 1)
            c = st.number_input("Introduce Total Recovered: ", 1, 1000000000, 5555, 1)
            d = st.number_input("Introduce Serious or Critical: ", 1, 100000000, 55, 1)
            e = st.number_input("Introduce Active Cases: ", 1, 100000000, 50000, 1)
            st.form_submit_button("Submit")

            model_users = numpy.array([a, b, c, d, e]).reshape([1, -1])
            users = predictie_concret(model_users)


            st.write('Total Cases prediction for ', f"*{teritoriul}*", 'is: ', round(users[-1]))


predictie_users()
