

import matplotlib.pyplot as plt # fixat prin versiunea 2.4.7 pyparsing
import streamlit as st
import yaml
from streamlit_authenticator import Authenticate

from footerul import footer
from Dataset_processing import modelarea, medie_modif





st.set_page_config(
        page_title="ML Methods Analysis: Basic graphs",
        page_icon="📈"
    )

def afisarea_grafice_concret():
    X_, y_ = modelarea()
    fig, ax = plt.subplots()
    boxplot = X_.boxplot(column=['Active Cases', 'Total Recovered'], ax=ax)
    st.write(fig)
    plt.savefig("boxplots_active_recovered.pdf", format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    X_, y_ = modelarea()
    hist = X_.hist(column=['Total Tests', 'Total Recovered', 'Active Cases', 'Population'], ax=ax)
    # plt.show()
    st.write(fig)
    plt.savefig("histogram_4columns.pdf", format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    X_, y_ = modelarea()
    X_["Total Recovered"].plot.density(color="red", ax=ax)
    plt.title('Density curve for the variable Total Recovered')
    st.write(fig)
    plt.savefig("density_curve.pdf", format="pdf", bbox_inches="tight")

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
    afisarea_grafice_concret()
    X_, _ = modelarea()
    l = ["-----"]
    for i in X_.columns:
        l.append(i)
    t = tuple(l)
    st.subheader("Column selection in displaying Boxplots, Histograms, Density Curves")
    coloana = st.selectbox("Select the column ", t)


    for i in X_.columns:
        if coloana==i:
            afisarea_grafice_general(col=i)

basic_graf()

##### Evaluation metrics KNN Model


from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, roc_auc_score

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn import tree
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, roc_curve, auc
import numpy as np
import pandas
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


# Citire si afisare date
df = pd.read_csv('septembrie2020_WM_dataset_augmentat.csv')
print(df.head(10).to_string(index=False))
df.head()
df.info()

# Tratare coloane nil
df.isnull().sum()
df = df.fillna(df.mean())
df.isnull().sum()
print(df.head())


# Vector de variabile independente si specificatie variabila dependenta
X = df.drop(['Country', 'Total Cases'], axis=1)
y = df['Total Cases']

print(X)
print(y)


def corelatie():
    # print("\nVerif corelatia Pearson dintre variabile si Total Cases")
    print("\nCorelatia intre variabilele independente si cea dependenta ")
    for i in X.columns:
        corelatie, _ = pearsonr(X[i], y)
        print(i + ': %.2f' % corelatie)
corelatie()


def medie_dispersie_devstd_Population():
    # medie coloana Population
    pavg = df["Population"].mean()
    # dispersie coloana Population
    pv = df["Population"].var()
    # deviatie standard coloana Population
    psd = df["Population"].std()
    # mediana coloana Population
    pmed = df["Population"].median()
    # cuartila coloana Population
    pq = df["Population"].quantile([0.25, 0.5, 0.75])
    return pavg, pv, psd, pmed, pq
medie_dispersie_devstd_Population()

def sumar_toate(col):
    # medie coloana
    avg = df[col].mean()
    # dispersie coloana
    v = df[col].var()
    # deviatie standard coloana
    sd = df[col].std()
    # mediana coloana
    med = df[col].median()
    # cuartile coloana
    q = df[col].quantile([0.25, 0.5, 0.75])
    return avg, v, sd, med, q
sumar_toate("Active Cases")

def matricea_heatmap_total():
    print("\nMatricea de relatie ")
    fig, ax = plt.subplots()
    mcorelatie = df.corr()
    sb.heatmap(mcorelatie, ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig("heatmap_ds2.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    # plt.show(fig)
matricea_heatmap_total()

def matricea_heatmap_var_ind():
    print("\nMatricea de relatie pentru potentiala variabila independenta ")
    fig, ax = plt.subplots()
    mcorelatieX = X.corr()
    sb.heatmap(mcorelatieX, ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.show()
    # plt.show(fig)
matricea_heatmap_var_ind()


X = df[['Total Tests', 'Total Recovered', 'Serious or Critical', 'Active Cases']]
print(X)


# Definire modele

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

ln = []

l_tc = []
for i in df.columns:
  if i == "Total Cases":
      l_tc.append(df[i])

l_po = []
for i in df.columns:
  if i == "Population":
    l_po.append(df[i])

for i in range(len(l_tc)):
    res = l_tc[i]/l_po[i]
    print(res)
    print(type(res))
    floatsolutie = pd.to_numeric(res, errors='coerce')

print(floatsolutie.mean())
for i in range(len(floatsolutie)):
    if floatsolutie[i] <= 0.003:
        ln.append(0)
    elif floatsolutie[i] > 0.003:
        ln.append(1)
print(ln)


df["Target"] = ln

y = df["Target"]

le = LabelEncoder()
y = le.fit_transform(y)

l_sensibilitate = [0]*5
l_specificitate = [0]*5
l_acuratete = [0]*5
l_precizie = [0]*5
l_scorulf1 = [0]*5


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = False)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# predict probabilities for the testing data
y_proba = knn.predict_proba(X_test)[:, 1]
# calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba)
# print the ROC AUC score
print("ROC AUC result:", roc_auc)
fpr, tpr, threshold1 = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of KNN')
plt.savefig("KNN-rocauc.pdf", format="pdf", bbox_inches="tight")
# plt.show()
st.write("KNN: ", fig)


scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knnc = KNeighborsClassifier(n_neighbors=10)
knnc.fit(X_train, y_train)
y_pred = knnc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12, 6))
plt.title("Confusion Matrix KNN")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig("KNN-cm.pdf", format="pdf", bbox_inches="tight")

# st.write(fig)

print(cm)
TP = cm[1][1]
# print(TP)
TN = cm[0][0]
# print(TN)
FP = cm[1][0]
FN = cm[0][1]

sensibilitate = TP/(TP+FN)
specificitate = TN/(TN+FP)
acuratete = (TP+TN)/(TP+TN+FP+FN)
precizie = TP/(TP+FP)
scorulf1 = TP/(TP+1/2*(FN+FP))

l_sensibilitate[0] = sensibilitate
l_specificitate[0] = specificitate
l_acuratete[0] = acuratete
l_precizie[0] = precizie
l_scorulf1[0] = scorulf1

print(l_sensibilitate)

print("\nKNN model ")
st.write("Sensibilitatea: ", sensibilitate)
st.write("Specificitatea: ", specificitate)
st.write("Acuratetea", acuratete)
st.write("Precizie: ", precizie)
st.write("Scorul F1", scorulf1)

knnlist = [sensibilitate, specificitate, acuratete, precizie, scorulf1]


##### Evaluation metrics Decision Tree Model

model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
model.fit(X_train, y_train)
# predict probabilities for the testing data
y_proba = model.predict_proba(X_test)[:, 1]
# calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba)
# print the ROC AUC score
print("ROC AUC result:", roc_auc)
fpr, tpr, threshold1 = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of DT')
plt.savefig("DT-rocauc.pdf", format="pdf", bbox_inches="tight")
st.write("DT: ", fig)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
dtc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=4)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix DT")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig("DT-cm.pdf", format="pdf", bbox_inches="tight")

print(cm)
TP = cm[1][1]
# print(TP)
TN = cm[0][0]
# print(TN)
FP = cm[1][0]
FN = cm[0][1]

sensibilitate = TP/(TP+FN)
specificitate = TN/(TN+FP)
acuratete = (TP+TN)/(TP+TN+FP+FN)
precizie = TP/(TP+FP)
scorulf1 = TP/(TP+1/2*(FN+FP))

l_sensibilitate[1] = sensibilitate
l_specificitate[1] = specificitate
l_acuratete[1] = acuratete
l_precizie[1] = precizie
l_scorulf1[1] = scorulf1

print(l_sensibilitate)

print("\nDecision Tree model ")
st.write("Sensibilitate: ", sensibilitate)
st.write("Specificitate: ", specificitate)
st.write("Acuratete: ", acuratete)
st.write("Precizie: ", precizie)
st.write("Scorul F1: ", scorulf1)

dtclist = [sensibilitate, specificitate, acuratete, precizie, scorulf1]


##### Evaluation metrics SVM Model

model = SVC(C=0.4, kernel="poly", degree=5, decision_function_shape="ovo", probability=True)
model.fit(X_train, y_train)
# predict probabilities for the testing data
y_proba = model.predict_proba(X_test)[:, 1]
# calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba)
# print the ROC AUC score
print("ROC AUC result:", roc_auc)
fpr, tpr, threshold1 = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of SVM')
plt.savefig("SVM-rocauc.pdf", format="pdf", bbox_inches="tight")
st.write("SVM: ", fig)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
svmc = SVC(C=0.5, kernel="poly", degree=5, decision_function_shape="ovo")
svmc.fit(X_train, y_train)
y_pred = svmc.predict(X_test)

cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix SVM")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig("SVM-cm.pdf", format="pdf", bbox_inches="tight")

print(cm)
TP = cm[1][1]
# print(TP)
TN = cm[0][0]
# print(TN)
FP = cm[1][0]
FN = cm[0][1]

sensibilitate = TP/(TP+FN)
specificitate = TN/(TN+FP)
acuratete = (TP+TN)/(TP+TN+FP+FN)
precizie = TP/(TP+FP)
scorulf1 = TP/(TP+1/2*(FN+FP))


l_sensibilitate[2] = sensibilitate
l_specificitate[2] = specificitate
l_acuratete[2] = acuratete
l_precizie[2] = precizie
l_scorulf1[2] = scorulf1

print(l_sensibilitate)

print("\nSVM model ")
st.write("Sensibilitate: ", sensibilitate)
st.write("Specificitate: ", specificitate)
st.write("Acuratete: ", acuratete)
st.write("Precizie: ", precizie)
st.write("Scorul F1: ", scorulf1)

svmclist = [sensibilitate, specificitate, acuratete, precizie, scorulf1]


##### Evaluation metrics CART Model


model = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
model.fit(X_train, y_train)
# predict probabilities for the testing data
y_proba = model.predict_proba(X_test)[:, 1]
# calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
# print the ROC AUC score
print("ROC AUC result:", roc_auc)
fpr, tpr, threshold1 = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of CART')
plt.savefig("CART-rocauc.pdf", format="pdf", bbox_inches="tight")
st.write("CART: ", fig)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
modelcart = tree.DecisionTreeClassifier(criterion="gini", max_depth=5)
modelcart.fit(X_train, y_train)
y_pred = modelcart.predict(X_test)

print(y_test)
print(y_pred)
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12, 6))
plt.title("Confusion Matrix CART")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig("CART-mse.pdf", format="pdf", bbox_inches="tight")

print(cm)
TP = cm[1][1]
# print(TP)
TN = cm[0][0]
# print(TN)
FP = cm[1][0]
FN = cm[0][1]

sensibilitate = TP/(TP+FN)
specificitate = TN/(TN+FP)
acuratete = (TP+TN)/(TP+TN+FP+FN)
precizie = TP/(TP+FP)
scorulf1 = TP/(TP+1/2*(FN+FP))


l_sensibilitate[3] = sensibilitate
l_specificitate[3] = specificitate
l_acuratete[3] = acuratete
l_precizie[3] = precizie
l_scorulf1[3] = scorulf1

print(l_sensibilitate)

print("\nCART model ")
st.write("Sensibilitate: ", sensibilitate)
st.write("Specificitate: ", specificitate)
st.write("Acuratete: ", acuratete)
st.write("Precizie: ", precizie)
st.write("Scorul F1: ", scorulf1)


xgboostlist = [sensibilitate, specificitate, acuratete, precizie, scorulf1]


##### Evaluation metrics XGBoost Model

model = XGBClassifier(subsample=0.15, max_depth=2)
model.fit(X_train, y_train)
# predict probabilities for the testing data
y_proba = model.predict_proba(X_test)[:, 1]
# calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
# print the ROC AUC score
print("ROC AUC result:", roc_auc)
fpr, tpr, threshold1 = roc_curve(y_test, y_proba)
fig, ax = plt.subplots()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of XGBoost')
plt.savefig("XGBoost-rocauc.pdf", format="pdf", bbox_inches="tight")
st.write("XGBoost: ", fig)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
xgc = XGBClassifier(subsample=0.15, max_depth=2)
xgc.fit(X_train, y_train)
y_pred = xgc.predict(X_test)

print(y_test)
print(y_pred)
cm=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(12,6))
plt.title("Confusion Matrix XGBoost")
sns.heatmap(cm, annot=True,fmt='d', cmap='Blues')
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.savefig("XGB-cm.pdf", format="pdf", bbox_inches="tight")

print(cm)
TP = cm[1][1]
# print(TP)
TN = cm[0][0]
# print(TN)
FP = cm[1][0]
FN = cm[0][1]

sensibilitate = TP/(TP+FN)
specificitate = TN/(TN+FP)
acuratete = (TP+TN)/(TP+TN+FP+FN)
precizie = TP/(TP+FP)
scorulf1 = TP/(TP+1/2*(FN+FP))

l_sensibilitate[4] = sensibilitate
l_specificitate[4] = specificitate
l_acuratete[4] = acuratete
l_precizie[4] = precizie
l_scorulf1[4] = scorulf1

print(l_sensibilitate)

print("\nXGBoost model ")
st.write("Sensibilitate: ", sensibilitate)
st.write("Specificitate: ", specificitate)
st.write("Acuratete: ", acuratete)
st.write("Precizie: ", precizie)
st.write("Scorul F1: ", scorulf1)

xgboostlist = [sensibilitate, specificitate, acuratete, precizie, scorulf1]





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


