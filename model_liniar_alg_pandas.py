
from math import sqrt

import numpy
from sklearn import tree
from sklearn.linear_model import LinearRegression

import pandas as pd
import streamlit as st

from prelucrare_date_pandas import modelarea



def modelul():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_) # fitting, potrivirea de date
    return model

def model_LR_coeffs():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_)
    r_sq = model.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r  # coeficientul de determinare si coeficientul de relatie


def predictie_concret(real):
    model = modelul()
    y_tara_tc = model.predict(real)
    return y_tara_tc

def predictie_general():
    model = modelul()
    X_, _ = modelarea()
    y_pred = model.predict(X_)
    df = pd.DataFrame(y_pred, columns=['Prediction value'])
    return df


def output_model():
    lr_r_sq, lr_r = model_LR_coeffs()
    st.write("Determination coefficient: ", lr_r_sq)
    st.write("Relation coefficient: ", lr_r)

    uk = numpy.array([68592949.0, 522526476.0, 22142505.0, 156.0, 348910.0]).reshape((1, -1))
    y_uk = predictie_concret(uk)
    st.write('Prediction of Total Cases in United Kingdom: ', round(y_uk[-1]))
    st.write('General prediction of Total Cases ', predictie_general())



