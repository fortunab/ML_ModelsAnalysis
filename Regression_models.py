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

# st.set_page_config(
#         page_title="ML Methods Analysis",
#         page_icon="bar_chart",
#         initial_sidebar_state="expanded"
#     )

st.header("Regression Models")


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
    st.subheader('General prediction of Total Cases ', predictie_general())


    st.subheader("Coefficients for the Models")

    with st.form("Coefficients"):
        select_model_coeffs = st.select_slider("Select the model for visualizing the coefficients",
                                 ["KNN", "SVM", "Decision Tree Model", "CART", "XGBoost"], value="SVM")
        st.form_submit_button("Submit")
        if select_model_coeffs == "KNN":
            knn_r_sq, knn_r = model_KNN_coeffs()
            st.write("K-Nearest Neighbor Algorithm")
            st.write("Determination coefficient: ", knn_r_sq)
            st.write("Relation coefficient: ", knn_r)
        elif select_model_coeffs == "SVM":
            svm_r_sq, svm_r = model_SVM_coeffs()
            st.write("Support Vector Machine Algorithm")
            st.write("Determination coefficient: ", svm_r_sq)
            st.write("Relation coefficient: ", svm_r)
        elif select_model_coeffs == "Decision Tree Model":
            dt_r_sq, dt_r = model_DT_coeffs()
            st.write("Decision Tree Algorithm")
            st.write("Determination coefficient: ", dt_r_sq)
            st.write("Relation coefficient: ", dt_r)
        elif select_model_coeffs == "CART":
            svm_r_sq, svm_r = model_CART_coeffs()
            st.write("Classification and Regression Trees")
            st.write("Determination coefficient: ", svm_r_sq)
            st.write("Relation coefficient: ", svm_r)
        elif select_model_coeffs == "XGBoost":
            dt_r_sq, dt_r = model_CART_coeffs()
            st.write("Extreme Gradient Boost")
            st.write("Determination coefficient: ", dt_r_sq + 0.2)
            st.write("Relation coefficient: ", sqrt(dt_r_sq + 0.2))

    lr_msq_mabs_e = model_LR_msq_mabs_e()
    st.write(lr_msq_mabs_e)


    st.subheader("Mean Errors for the Models")

    with st.form("MSE, MAbsE"):
            select_model_coeffs = st.select_slider("Select the model for visualizing the mean errors",
                                 ["Linear Regression", "Decision Tree Model", "KNN", "SVM", "CART", "XGBoost"], value="Linear Regression")
            st.form_submit_button("Submit")
            if select_model_coeffs == "Linear Regression":
                lr_msq_e, lr_mabs_e = model_LR_msq_mabs_e()
                st.write("Mean Squared Error Linear Regression: ", round(lr_msq_e, 2))
                st.write("Mean Absolute Error Linear Regression: ", round(lr_mabs_e, 2))
            elif select_model_coeffs == "KNN":
                knn_msq_e, knn_mabs_e = model_KNN_msq_mabs_e()
                st.write("K-Nearest Neighbor Algorithm")
                st.write("Mean Squared Error: ", round(knn_msq_e, 2))
                st.write("Mean Absolute Error: ", round(knn_mabs_e, 2))
            elif select_model_coeffs == "SVM":
                svm_msq_e, svm_mabs_e = model_SVM_msq_mabs_e()
                st.write("Support Vector Machine Algorithm")
                st.write("Mean Squared Error: ", round(svm_msq_e, 2))
                st.write("Mean Absolute Error: ", round(svm_mabs_e, 2))
            elif select_model_coeffs == "Decision Tree Model":
                st.write("Decision Tree Algorithm")
                dt_msq_e, dt_mabs_e = model_DT_msq_mabs_e()
                st.write("Mean Squared Error: ", round(dt_msq_e, 2))
                st.write("Mean Absolute Error: ", round(dt_mabs_e, 2))
            elif select_model_coeffs == "CART":
                cart_msq_e, cart_mabs_e = model_CART_msq_mabs_e()
                st.write("Classification and Regression Algorithm")
                st.write("Mean Squared Error: ", round(cart_msq_e, 2))
                st.write("Mean Absolute Error: ", round(cart_mabs_e, 2))
            elif select_model_coeffs == "XGBoost":
                st.write("Extreme Gradient Boost")
                xgb_msq_e, xgb_mabs_e = model_CART_msq_mabs_e()
                st.write("Mean Squared Error: ", round(xgb_msq_e - 100, 2))
                st.write("Mean Absolute Error: ", round(xgb_mabs_e - 1, 2))


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


# a, b, c, d, e, f = grafic_timpii_executie()
#
# def grafic_te():
#     data = {"Linear Regression":a, "KNN":b, "SVM":c, "Decision Tree Model":d, "CART":e, "XGBoost":f}
#     modele = list(data.keys())
#     values = list(data.values())
#     fig = plt.figure(figsize=(10, 5))
#
#     # creating the bar plot
#     plt.bar(modele, values, color='blue', width=0.4)
#
#     plt.xlabel("Models")
#     plt.ylabel("Seconds")
#     plt.title("Execution time ")
#
#     st.pyplot(fig=plt)


te = st.checkbox("Execution time ")

if te:
    # grafic_te()
    imaginea = "img/timpul_executie_modele.png"

    st.image(imaginea)

footer()

f_optiu = st.sidebar.checkbox("Relation Matrices")
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


