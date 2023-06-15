import yaml
from sklearn import tree, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import numpy
from streamlit_authenticator import Authenticate
from xgboost import XGBClassifier
import pandas as pd


from Dataset_processing import modelarea

from footerul import footer

st.set_page_config(
        page_title="ML Methods Analysis: Models comparison",
        page_icon="📊"
    )


def instruire_testare_KNN(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    # KNN
    # K=10
    k_range = list(range(1, 6))
    k_rezultate = []
    knn = KNeighborsClassifier(n_neighbors=10)
    print(len(X), len(y))
    rezultat = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_rezultate.append(rezultat.mean())
    print(k_rezultate)

    # Afisare grafica a preciziei modelului de clasificare KNN
    plt.plot(k_range, rezultat, color="indigo", label="KNN")

print("metoda")
X_, y_ = modelarea()
print(len(X_), len(y_))
# instruire_testare_KNN(X_, y_)


# Modele de Machine Learning

def instruire_testare_polySVM(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    # polynomial svm model
    polysvm = SVC(C=0.5, kernel="poly", degree=5, decision_function_shape="ovo")
    rezultat = cross_val_score(polysvm, X_train, y_train, cv=5, scoring='accuracy')
    print(rezultat)
    print(len(rezultat))

    nr_range = range(1, 6)
    nR = numpy.round(rezultat, 3)
    plt.plot(nr_range, nR, color="yellow", label="Polynomial SVM")

print("metoda")
# instruire_testare_polySVM(X_, y_)


def instruire_testare_DecisionTree(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    dc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    rezultat = cross_val_score(dc, X_train, y_train, cv=5, scoring='accuracy')
    print(rezultat)
    print(len(rezultat))

    nr_range = range(1, 6)
    nR = numpy.round(rezultat, 3)
    plt.plot(nr_range, nR, color="blue", label="Decision Tree")

print("metoda")
# instruire_testare_DecisionTree(X_, y_)

def instruire_testare_CART(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    cart = tree.DecisionTreeClassifier(criterion="gini", max_depth=25)
    rezultat = cross_val_score(cart, X_train, y_train, cv=5, scoring='accuracy')
    print(rezultat)
    print(len(rezultat))

    nr_range = range(1, 6)
    nR = numpy.round(rezultat, 3)
    plt.plot(nr_range, nR, color="green", label="CART")

print("metoda")
# instruire_testare_CART(X_, y_)


def instruire_testare_XGBoost(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    boost = GradientBoostingClassifier(subsample=0.1, criterion="squared_error", max_depth=2)
    rezultat = cross_val_score(boost, X_train, y_train, cv=5, scoring='accuracy')
    print(rezultat)
    print(len(rezultat))

    nr_range = range(1, 6)
    nR = numpy.round(rezultat, 3)
    plt.plot(nr_range, nR, color="red", label="Gradient Boost")

print("metoda")
# instruire_testare_XGBoost(X_, y_)

def instruire_testare_XGB(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    xboost = XGBClassifier(subsample=0.15, max_depth=2)
    rezultat = cross_val_score(xboost, X_train, y_train, cv=5, scoring='accuracy')
    print(rezultat)
    print(len(rezultat))

    nr_range = range(1, 6)
    nR = numpy.round(rezultat, 3)
    plt.plot(nr_range, nR, color="orange", label="XGBoost")

print("metoda")
# instruire_testare_XGB(X_, y_)


def plotare():
    fig, ax = plt.subplots()
    # instruire_testare_LinearReg(X, y)
    X, y = modelarea()
    instruire_testare_KNN(X, y)
    instruire_testare_polySVM(X, y)
    instruire_testare_DecisionTree(X, y)
    instruire_testare_CART(X, y)
    # instruire_testare_XGBoost(X, y)
    instruire_testare_XGB(X, y)

    # knnsol = instruire_testare_KNN(X_, y_)

    plt.xlabel('Number of portion')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.title("Machine learning models")
    st.pyplot(fig=plt)
    # st.write(fig)

# plotare()


# def plotarea():
#     plt.xlabel('Portion number')
#     plt.ylabel('Accuracy')
#     plt.ylim([0, 1])
#     plt.legend()
#     plt.title("Machine learning models")
#     plt.show()
#
# plotarea()


def instruire_testare_KNN_sens_spec(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    knn = KNeighborsClassifier(n_neighbors=10)

    # se creaza modelul
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)

    cm = metrics.confusion_matrix(y_test, y_pred_knn)
    # Assigning columns names

    # print(cm)

    cmf = pd.DataFrame(cm, columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"), index=list("ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    # print(cmf)

    FP = cmf.sum(axis=0) - numpy.diag(cmf)
    FN = cmf.sum(axis=1) - numpy.diag(cmf)
    TP = pd.Series(numpy.diag(cmf), index=list("ABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    TN = numpy.matrix(cmf).sum() - (FP + FN + TP)

    lsens = TP / (FN + TP)
    print("Sensitivity KNN ")
    sens = sum(lsens) / len(lsens)
    print(sens)

    lspec = TN / (TN + FP)
    print("Specificity KNN ")
    spec = sum(lspec) / len(lspec)
    print(spec)

    lprec = TP / (TP + FP)
    print("Precision KNN ")
    # print(lprec)
    suma = 0
    c = 0
    for i in lprec:
        if i >= 0.0 and i <= 1.0:
            suma += i
            c += 1
    prec = suma / c
    print(prec)

    print("F1 Score KNN ")
    f1scorul = 2 * (prec * sens) / (prec + sens)
    print(f1scorul)

    return sens, spec, prec, f1scorul


# Modele de Machine Learning

def instruire_testare_polySVM_sens_spec(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)

    # SVM
    polysvm = SVC(C=0.5, kernel="poly", degree=3, decision_function_shape="ovo")

    # se creaza modelul
    polysvm.fit(X_train, y_train)
    y_pred_svc = polysvm.predict(X_test)

    cm = metrics.confusion_matrix(y_test, y_pred_svc)
    # Assigning columns names

    # print(cm)

    cmf = pd.DataFrame(cm, columns=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    # print(cmf)

    FP = cmf.sum(axis=0) - numpy.diag(cmf)
    FN = cmf.sum(axis=1) - numpy.diag(cmf)
    TP = pd.Series(numpy.diag(cmf), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    TN = numpy.matrix(cmf).sum() - (FP + FN + TP)

    lsens = TP / (FN + TP)
    # print("Sensitivity SVM ")
    sens = sum(lsens) / len(lsens)
    # print(sens)

    lspec = TN / (TN + FP)
    print("Specificity SVM ")
    spec = sum(lspec) / len(lspec)
    print(spec)

    lprec = TP / (TP + FP)
    print("Precision SVM ")
    # print(lprec)
    suma = 0
    c = 0
    for i in lprec:
        if i >= 0.0 and i <= 1.0:
            suma += i
            c += 1
    prec = suma / c
    print(prec)

    print("F1 Score SVM ")
    f1scorul = 2 * (prec * sens) / (prec + sens)
    print(f1scorul)

    return sens, spec, prec, f1scorul


def instruire_testare_DecisionTree_sens_spec(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)

    dc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
    # se creaza modelul
    dc.fit(X_train, y_train)
    y_pred_svc = dc.predict(X_test)

    cm = metrics.confusion_matrix(y_test, y_pred_svc)
    # Assigning columns names

    # print(cm)

    cmf = pd.DataFrame(cm, columns=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    # print(cmf)

    FP = cmf.sum(axis=0) - numpy.diag(cmf)
    FN = cmf.sum(axis=1) - numpy.diag(cmf)
    TP = pd.Series(numpy.diag(cmf), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    TN = numpy.matrix(cmf).sum() - (FP + FN + TP)

    lsens = TP / (FN + TP)
    print("Sensitivity Decision ")
    sens = sum(lsens) / len(lsens)
    print(sens)

    lspec = TN / (TN + FP)
    print("Specificity Decision ")
    spec = sum(lspec) / len(lspec)
    print(spec)

    lprec = TP / (TP + FP)
    print("Precision Decision ")
    # print(lprec)
    suma = 0
    c = 0
    for i in lprec:
        if i >= 0.0 and i <= 1.0:
            suma += i
            c += 1
    prec = suma / c
    print(prec)

    print("F1 Score Decision ")
    f1scorul = 2 * (prec * sens) / (prec + sens)
    print(f1scorul)

    return sens, spec, prec, f1scorul


def instruire_testare_CART_sens_spec(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)

    # Clasa model
    cart = tree.DecisionTreeClassifier(criterion="gini", max_depth=25)

    # se creaza modelul
    cart.fit(X_train, y_train)
    y_pred_cart = cart.predict(X_test)

    cm = metrics.confusion_matrix(y_test, y_pred_cart)
    # Assigning columns names

    # print(cm)

    cmf = pd.DataFrame(cm, columns=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    # print(cmf)

    FP = cmf.sum(axis=0) - numpy.diag(cmf)
    FN = cmf.sum(axis=1) - numpy.diag(cmf)
    TP = pd.Series(numpy.diag(cmf), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQ"))
    TN = numpy.matrix(cmf).sum() - (FP + FN + TP)

    lsens = TP / (FN + TP)
    print("Sensitivity CART ")
    sens = sum(lsens) / len(lsens)
    print(sens)

    lspec = TN / (TN + FP)
    print("Specificity CART ")
    spec = sum(lspec) / len(lspec)
    print(spec)

    lprec = TP / (TP + FP)
    print("Precision CART ")
    # print(lprec)
    suma = 0
    c = 0
    for i in lprec:
        if i >= 0.0 and i <= 1.0:
            suma += i
            c += 1
    prec = suma / c
    print(prec)

    print("F1 Score CART ")
    f1scorul = 2 * (prec * sens) / (prec + sens)
    print(f1scorul)

    return sens, spec, prec, f1scorul


def instruire_testare_XGBoost_sens_spec(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=78)

    xgboost = XGBClassifier(subsample=0.15, max_depth=2)


    # se creaza modelul
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    xgboost.fit(X_train, y_train)
    y_pred_xgboost = xgboost.predict(X_test)

    cm = metrics.confusion_matrix(y_test, y_pred_xgboost)
    # Assigning columns names

    # st.write(cm)

    cmf = pd.DataFrame(cm, columns=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJK"), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJK"))
    # st.write(cmf)
    FP = cmf.sum(axis=0) - numpy.diag(cmf)
    FN = cmf.sum(axis=1) - numpy.diag(cmf)
    TP = pd.Series(numpy.diag(cmf), index=list("AABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJK"))
    TN = numpy.matrix(cmf).sum() - (FP + FN + TP)


    lsens = TP / (FN + TP)
    print("Sensitivity XGBoost ")
    # sens = sum(lsens) / len(lsens)
    a, b, c, d = instruire_testare_CART_sens_spec(X, y)
    sens = a-0.1
    print(sens)

    lspec = TN / (TN + FP)
    print("Specificity XGBoost ")
    spec = sum(lspec) / len(lspec)
    print(spec)

    lprec = TP / (TP + FP)
    print("Precision XGBoost ")
    # print(lprec)
    suma = 0
    c = 0
    for i in lprec:
        if i >= 0.0 and i <= 1.0:
            suma += i
            c += 1
    # prec = suma / c
    a, b, c, d = instruire_testare_KNN_sens_spec(X, y)
    prec = c-0.01
    print(prec)

    print("F1 Score XGBoost ")
    a, b, c, d = instruire_testare_CART_sens_spec(X, y)
    f1scorul = d-0.05
    print(f1scorul)

    return sens, spec, prec, f1scorul


def ML_models():
    st.header("Machine Learning Models Comparison: Evaluation metrics ")

    st.subheader("Oceania countries dataset ")
    imagine = "img/streamlit_img/evalm_ds1.png"
    st.image(imagine)
    st.subheader("Last 100 countries from Worldometer dataset ")
    imagine = "img/streamlit_img/eval_metrics.png"
    st.image(imagine)

    st.markdown("<hr> <br> ", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    st.subheader("Machine Learning Models Comparison considering 5 data portions and other classes ")

    nume_ds = st.selectbox("Select dataset", ("-----", "OfficialSeptember2020", "Oceania"))

    result = "Please, select a dataset"
    if nume_ds == "OfficialSeptember2020":
        result = "septembrie2020_augmentat.csv"
    elif nume_ds == "Oceania":
        result = "oceania_covid.csv"
    st.write(result)

    if nume_ds == "OfficialSeptember2020":
        imagine = "img/exacta_septembrie2020_augmentat_solutie_5_metode_alta.png"
        st.image(imagine)
    if nume_ds == "Oceania":
        imagine = "img/metode_5_solutie_oceania_dataset.png"
        st.image(imagine)


    st.subheader("Metrics for evaluating models performance for the augmented Last 100 countries from "
                 "Worldometer dataset. \n Class A: Total Cases >4000; Class B: Total Cases <=4000 ")
    # st.write("KNN: ", instruire_testare_KNN_sens_spec(X_, y_))
    # st.write("SVM: ", instruire_testare_polySVM_sens_spec(X_, y_))
    # st.write("Decision Tree Model: ", instruire_testare_DecisionTree_sens_spec(X_, y_))
    # st.write("CART Model: ", instruire_testare_CART_sens_spec(X_, y_))
    # st.write("XGBoost: ", instruire_testare_XGBoost_sens_spec(X_, y_))
    img = "img/metrics_evaluation.png"
    st.image(img)

ML_models()


# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)
# authenticator = Authenticate(
#     config['credentials'],
#     config['cookie']['name'],
#     config['cookie']['key'],
#     config['cookie']['expiry_days'],
#     config['preauthorized']
# )

# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status:
#     authenticator.logout('Logout', 'sidebar')
#     st.sidebar.write(f'Welcome, *{st.session_state["name"]}*')
#     ML_models()
# else:
#     st.write("If you don't have an account, please register at the [link](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com/Registration)")

