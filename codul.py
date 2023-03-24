# linia 444 in ML_models_comparison.py s-a folosit
# knnsens, knnspec, knnprec, knnscorulf1 = instruire_testare_KNN_sens_spec(X_, y_)
# svmsens, svmspec, svmprec, svmscorulf1 = instruire_testare_polySVM_sens_spec(X_, y_)
# dcsens, dcspec, dcprec, dcscorulf1 = instruire_testare_DecisionTree_sens_spec(X_, y_)
# cartsens, cartspec, cartprec, cartscorulf1 = instruire_testare_CART_sens_spec(X_, y_)
# xgsens, xgspec, xgprec, xgbscorulf1 = instruire_testare_XGBoost_sens_spec(X_, y_)

"""
def sensibilitatea():
    data = {"KNN":knnsens, "SVM":svmsens, "Decision Tree Model":dcsens, "CART":cartsens, "XGBoost":xgsens}
    modele = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))

    # creating the bar chart plot
    plt.bar(modele, values, color='blue',
            width=0.4)

    plt.xlabel("Models")
    plt.ylabel("Sensitivity")
    plt.title("Individual Sensitivity of the Models")

    st.pyplot(fig=plt)
    # st.write(fig)

def specificitatea():
    data = {"KNN":knnspec, "SVM":svmspec, "Decision Tree Model":dcspec, "CART":cartspec, "XGBoost":xgspec}
    modele = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(modele, values, color='yellow',
            width=0.4)

    plt.xlabel("Models")
    plt.ylabel("Specificity")
    plt.title("Individual Specificity of the Models")

    st.pyplot(fig=plt)
    # st.write(fig)

def precizia():
    data = {"KNN":knnprec, "SVM":svmprec, "Decision Tree Model":dcprec, "CART":cartprec, "XGBoost":xgprec}
    modele = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(modele, values, color='green',
            width=0.4)

    plt.xlabel("Models")
    plt.ylabel("Precision")
    plt.title("Individual Precision of the Models")

    st.pyplot(fig=plt)
    # st.write(fig)

def scorulF1():
    data = {"KNN":knnscorulf1, "SVM":svmscorulf1, "Decision Tree Model":dcscorulf1, "CART":cartscorulf1, "XGBoost":xgbscorulf1}
    modele = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))

    # creating the bar chart
    plt.bar(modele, values, color='red',
            width=0.4)

    plt.xlabel("Models")
    plt.ylabel("F1 Score")
    plt.title("Individual F1 Score of the Models")

    st.pyplot(fig=plt)
    # st.write(fig)
"""

