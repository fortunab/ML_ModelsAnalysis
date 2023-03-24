from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import streamlit as st
import numpy
from xgboost import XGBClassifier

from prelucrare_date_pandas import modelarea

fig, ax = plt.subplots()



def instruire_testare_KNN(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    # KNN
    # K=10
    k_range = list(range(1, 6))
    k_rezultate = []
    # for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=10)
    print(len(X), len(y))
    rezultat = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_rezultate.append(rezultat.mean())
    print(k_rezultate)

    # Afisare grafica a preciziei modelului de clasificare KNN
    # in conformitate cu numerele de acces (GSM) ale genei
    plt.plot(k_range, rezultat, color="indigo", label="KNN")

print("metoda")
X_, y_ = modelarea()
print(len(X_), len(y_))
instruire_testare_KNN(X_, y_)


# Modele de Machine Learning

def instruire_testare_polySVM(X, y):
    # Instruire si testare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=False)

    # regresia logistica
    polysvm = SVC(C=0.5, kernel="poly", degree=5, decision_function_shape="ovo")
    rezultat = cross_val_score(polysvm, X_train, y_train, cv=5, scoring='accuracy')
    print(rezultat)
    print(len(rezultat))

    nr_range = range(1, 6)
    nR = numpy.round(rezultat, 3)
    plt.plot(nr_range, nR, color="yellow", label="Polynomial SVM")

print("metoda")
instruire_testare_polySVM(X_, y_)


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
instruire_testare_DecisionTree(X_, y_)

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
instruire_testare_CART(X_, y_)


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
instruire_testare_XGB(X_, y_)


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

