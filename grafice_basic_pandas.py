
import matplotlib.pyplot as plt # fixat prin versiunea 2.4.7 pyparsing
import streamlit as st

from prelucrare_date_pandas import modelarea


def afisarea_grafice_concret():
    X_, y_ = modelarea()
    fig, ax = plt.subplots()
    boxplot = X_.boxplot(column=['Active Cases', 'Total Recovered'], ax=ax)
    st.write(fig)

    fig, ax = plt.subplots()
    X_, y_ = modelarea()
    hist = X_.hist(column=['Total Tests', 'Total Recovered', 'Active Cases', 'Population'], ax=ax)
    # plt.show()
    st.write(fig)

    fig, ax = plt.subplots()
    X_, y_ = modelarea()
    X_["Total Recovered"].plot.density(color="red", ax=ax)
    plt.title('Density curve for the variable Total Recovered')
    st.write(fig)

def afisarea_grafice_general(col):
    X_, y_ = modelarea()
    fig, ax = plt.subplots()
    boxplot = X_.boxplot(col, ax=ax)
    st.write(fig)

    fig, ax = plt.subplots()
    X_, y_ = modelarea()
    hist = X_.hist(col, ax=ax)
    st.write(fig)

    fig, ax = plt.subplots()
    X_, y_ = modelarea()
    X_[col].plot.density(color="blue", ax=ax)
    plt.title('Density curve for the variable ' + col)
    st.write(fig)
