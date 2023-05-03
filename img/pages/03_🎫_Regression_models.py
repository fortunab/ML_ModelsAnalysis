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

st.markdown("<h1> Regression Model considering the augmented <i>Last 100 countries from Worldometer</i> dataset (DS2) </h1>", unsafe_allow_html=True)


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

def model_LR_msq_mabs_e():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_)
    y_pred = model.predict(X_)
    mse = mean_squared_error(y_, y_pred)
    mabs = mean_absolute_error(y_, y_pred)
    return mse, mabs

def model_DT_coeffs():
    dc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
    X_, y_ = modelarea()
    dc = dc.fit(X_, y_)
    r_sq = dc.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_DT_msq_mabs_e():
    model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    mabs = mean_absolute_error(y_pred, y_test)
    return mse, mabs

def model_SVM_coeffs():
    modelsvc = SVC(C=0.5, kernel="poly", degree=3, decision_function_shape="ovo")
    X_, y_ = modelarea()
    modelsvc = modelsvc.fit(X_, y_)
    r_sq = modelsvc.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_SVM_msq_mabs_e():
    model = SVC(C=0.5, kernel="poly", degree=5, decision_function_shape="ovo")
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    mabs = mean_absolute_error(y_pred, y_test)
    return mse, mabs

def model_KNN_coeffs():
    modelknn = KNeighborsClassifier(n_neighbors=10)
    X_, y_ = modelarea()
    modelknn = modelknn.fit(X_, y_)
    r_sq = modelknn.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_KNN_msq_mabs_e():
    model = KNeighborsClassifier(n_neighbors=10)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    mabs = mean_absolute_error(y_pred, y_test)
    return mse, mabs

def model_CART_coeffs():
    cart = tree.DecisionTreeClassifier(criterion="gini", max_depth=25)
    X_, y_ = modelarea()
    cart = cart.fit(X_, y_)
    r_sq = cart.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_CART_msq_mabs_e():
    model = tree.DecisionTreeClassifier(criterion="gini", max_depth=25)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    mabs = mean_absolute_error(y_pred, y_test)
    return mse, mabs

def model_XGBoost_coeffs():
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # y_train = le.fit_transform(y_train)

    # xboost = XGBClassifier(subsample=0.15, max_depth=2)
    # rezultat = cross_val_score(xboost, X_train, y_train, cv=5, scoring='accuracy')
    xgb = XGBClassifier(subsample=0.15, max_depth=2)
    X_, y_ = modelarea()
    y_ = le.fit_transform(y_)
    xgb = xgb.fit(X_, y_)
    r_sq = xgb.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_XGBoost_msq_mabs_e():
    model = XGBClassifier(subsample=0.15, max_depth=2)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_pred, y_test)
    mabs = mean_absolute_error(y_pred, y_test)
    return mse, mabs

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
    st.subheader("Linear Regression Algorithm Example")
    st.write("Determination coefficient: ", lr_r_sq)
    st.write("Relation coefficient: ", lr_r)

    uk = numpy.array([68592949.0, 522526476.0, 22142505.0, 156.0, 348910.0]).reshape((1, -1))
    y_uk = predictie_concret(uk)
    st.write('Prediction of Total Cases in United Kingdom: ', round(y_uk[-1]))



def timpii_executie():
    start_time = time.time()
    model_LR_coeffs()
    linr = "Linear Regression --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_KNN_coeffs()
    knn = "KNN --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_SVM_coeffs()
    polysvm = "SVM --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_DT_coeffs()
    dc = "Decision Tree Model --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_CART_coeffs()
    cart = "CART --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_XGBoost_coeffs()
    xgb = "XGB --- %s seconds ---" % (time.time() - start_time)


    return linr, knn, polysvm, dc, cart, xgb

# a, b, c, d, e, f = timpii_executie()
# for i in timpii_executie():
#     st.write(i)

def grafic_timpii_executie():
    start_time = time.time()
    model_LR_coeffs()
    linr = time.time() - start_time

    start_time = time.time()
    model_KNN_coeffs()
    knn = time.time() - start_time

    start_time = time.time()
    model_SVM_coeffs()
    polysvm = time.time() - start_time

    start_time = time.time()
    model_DT_coeffs()
    dc = time.time() - start_time

    start_time = time.time()
    model_CART_coeffs()
    cart = time.time() - start_time

    start_time = time.time()
    model_XGBoost_coeffs()
    xgb = time.time() - start_time


    return linr, knn, polysvm, dc, cart, xgb


a, b, c, d, e, f = grafic_timpii_executie()

def grafic_te():
    data = {"Linear Regression":a, "KNN":b, "SVM":c, "Decision Tree Model":d, "CART":e, "XGBoost":f}
    modele = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(modele, values, color='blue', width=0.4)

    plt.xlabel("Models")
    plt.ylabel("Seconds")
    plt.title("Execution time ")

    plt.savefig("timp_executieDS2.pdf", format="pdf", bbox_inches="tight")
    st.pyplot(fig=plt)
# grafic_te()
te = st.checkbox("Execution time ")

if te:
    # grafic_te()
    imaginea = "img/timpul_executie_modele.png"

    st.image(imaginea)


f_optiu = st.sidebar.checkbox("Relation Matrices ")
if f_optiu:
    matricea_heatmap()
    matricea_heatmap_var_ind()
output_model()

st.header("Manual input for concrete data, Total Cases prediction ")
def predictie_users():
    with st.form("predictie"):
            teritoriul = st.text_input("Introduce Country/Others name: ", "Default")
            a = st.number_input("Introduce population: ", 1, 100000000000, 40000, 1)
            b = st.number_input("Introduce Total Tests: ", 1, 100000000000, 500000, 1)
            c = st.number_input("Introduce Total Recovered: ", 1, 1000000000, 5555, 1)
            d = st.number_input("Introduce Serious or Critical: ", 1, 100000000, 55, 1)
            e = st.number_input("Introduce Active Cases: ", 1, 100000000, 50000, 1)
            st.form_submit_button("Submit")

            model_users = numpy.array([a, b, c, d, e]).reshape([1, -1])
            users = predictie_concret(model_users)


            st.write('Total Cases prediction for ', f"*{teritoriul}*", 'is: ', round(users[-1]))


predictie_users()

feedback = st.checkbox("Feedback")
if feedback:
    with st.form("Feedback"):
            st.header("Feedback")
            val = st.selectbox("How was your experience of this application?", ["-----", "Good", "Neutral", "Bad"])
            st.select_slider("How would you rate the application",
                             ["Poor", "Not Good", "As Expected", "Easy for follow", "Excellent"], value="As Expected")
            st.form_submit_button("Submit")
            if val != "-----":
                st.text("Thank you for your implication and for the feedback ")

# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)
# authenticator = Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )
#
# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status:
#     authenticator.logout('Logout', 'sidebar')
#     st.sidebar.write(f'Welcome, *{st.session_state["name"]}*')
#     modele()
# else:
#     st.write("If you don't have an account, please register at the [link](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com/Registration)")


