

import matplotlib.pyplot as plt # fixat prin versiunea 2.4.7 pyparsing
import streamlit as st
import yaml
from streamlit_authenticator import Authenticate

from footerul import footer
from Dataset_processing import modelarea, medie_modif





# st.set_page_config(
#         page_title="ML Methods Analysis",
#         page_icon="bar_chart",
#         initial_sidebar_state="expanded"
#     )

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
    X, _ = modelarea()
    ds = medie_modif()
    # for i in col:
    #     if i == max(col):
    #         for j in range(len(ds.select(col))):
    #             if i == col[j]:
    #                 st.write("Country/Others ", ds["Country"][j], "has the most ", col)

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

def basic_graf():

    st.header("Display basic graphs")

    footer()
    afisarea_grafice_concret()
    X_, _ = modelarea()
    l = ["-----"]
    for i in X_.columns:
        l.append(i)
    t = tuple(l)
    coloana = st.selectbox("Select the column ", t)


    for i in X_.columns:
        if coloana==i:
            afisarea_grafice_general(col=i)

basic_graf()

# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)
#
# authenticator = Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
#
#
# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status:
#     authenticator.logout('Logout', 'sidebar')
#     st.sidebar.write(f'Welcome, *{st.session_state["name"]}*')
#     basic_graf()
# else:
#     st.write("If you don't have an account, please register at the [link](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com/Registration)")


