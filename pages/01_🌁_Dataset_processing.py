
import matplotlib.pyplot as plt
import numpy
import seaborn as sb
import streamlit as st
import yaml
from scipy.stats import pearsonr
from streamlit_authenticator import Authenticate

from footerul import footer
from input_pandas import citire_file



st.set_page_config(
        page_title="ML Methods Analysis: Dataset processing",
        page_icon="🌁",
    )

def medie_modif():
    cf = citire_file()
    # se afiseaza media tuturor coloanelor numerice
    # valorile NaN se inlocuiesc cu media valorilor
    # de pe coloana respectiva, pentru fiecare coloana
    ult = cf.fillna(cf.mean(numeric_only=True))
    return ult

def medie_total_teste():
    mcl = medie_modif()
    medie_teste = mcl["Total Tests"].mean()
    medie_teste = "Total Tests mean: " + str(medie_teste)
    return medie_teste

def tara_totalcases():
    cf = medie_modif()
    # se afiseaza primele 10 observatii pentru relatia Country si Total Cases
    nume_casesTotal = cf[["Country", "Total Cases"]]
    return nume_casesTotal

def medie_dispersie_devstd_totalrec():
    cf = medie_modif()
    # medie coloana Total Recovered
    travg = str(round(cf["Total Recovered"].mean(), 2))
    # dispersie coloana Total Recovered
    trv = str(round(cf["Total Recovered"].var(), 2))
    # deviatie standard coloana Total Recovered
    trsd = str(round(cf["Total Recovered"].std(), 2))
    # mediana coloana Total Recovered
    trmed = str(round(cf["Total Recovered"].median(), 2))
    # cuartila coloana Total Recovered
    trq = str(round(cf["Total Recovered"].quantile([0.25, 0.5, 0.75])))
    return travg, trv, trsd, trmed, trq

def sumar_toate(col):
    cf = medie_modif()
    # medie coloana
    avg = round(cf[col].mean(), 2)
    # dispersie coloana
    v = round(cf[col].var(), 2)
    # deviatie standard coloana
    sd = round(cf[col].std(), 2)
    # mediana coloana
    med = round(cf[col].median(), 2)
    # cuartile coloana
    q = round(cf[col].quantile([0.25, 0.5, 0.75]), 2)
    return avg, v, sd, med, q

# Modelare date: X = predictor, var independenta, y = raspuns, var dependenta

# Pregatire date pentru modelul liniar
def modelarea():
    cf = medie_modif()
    X = cf.drop(columns=["Country", "Total Cases", "Total Deaths"], axis=1)
    y = cf["Total Cases"]
    return X, y

def corelatie():
    # print("\nVerif corelatia Pearson dintre variabile si Total Cases")
    X, y = modelarea()
    st.subheader("\nCorrelation between the variables and Total Cases variable ")
    for i in X.columns:
        corelatie, _ = pearsonr(X[i], y)
        st.write(i + ': %.2f' % corelatie)


def new_modelarea():
    X_, y_ = modelarea()
    X_ = X_.drop("Total Tests")
    return X_, y_

def matricea_heatmap():
    cf = medie_modif()
    fig, ax = plt.subplots()
    mcorelatie = cf.corr()
    sb.heatmap(mcorelatie, ax=ax)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    st.write(fig)

def matricea_heatmap_var_ind():
    print("Relation Matrix")
    X_, _ = modelarea()
    fig, ax = plt.subplots()
    mcorelatieX = X_.corr()
    sb.heatmap(mcorelatieX, ax=ax)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    # plt.show()
    st.write(fig)


def afisarea_prelucrarea():
    cf = medie_modif()
    cf.printSchema()
    cf.show(10)
    cf.groupBy(cf['Total Cases'] > 4000).count().show(10)
    medie_total_teste()
    tara_totalcases().head(10)
    X, y = modelarea()
    X.show()
    y.show()

    corelatie()
    matricea_heatmap()
    matricea_heatmap_var_ind()

def dataset_proc():

    st.markdown("<h2> Data Processing for the augmented <i> Last 100 countries from Worldometer until september 2020 </i> dataset (DS2) </h2> ", unsafe_allow_html=True)

    # nume_ds = st.selectbox("Select dataset", ("-----", "OfficialSeptember2020"))
    # if nume_ds == "OfficialSeptember2020":
    #     st.write(nume_ds + "DateSeptembrie2020_augmentat.csv")

    optiune1 = st.sidebar.checkbox("Initial dataset")
    optiune2 = st.sidebar.checkbox("Dataset with valid values")

    ds = citire_file()
    if optiune1:
        st.write(ds)

    cf = medie_modif()
    if optiune2:
        st.write(cf)

    # def get_dataset(nume_ds):
    #     if nume_ds == "DateSeptembrie2020":
    #         X_, y_ = modelarea()
    #     else:
    #         X_, y_ = modelarea()
    #     return X_, y_

    X, y = modelarea()

    optiune3 = st.checkbox("Prediction variable")
    if optiune3:
        st.write(X)
        st.write("Number of observations and columns: ", X.shape)

    optiune4 = st.checkbox("Response variable")
    if optiune4:
        st.write(y)
        st.write("Total number of independent countries from the list: ", len(numpy.unique(y)))

    optiune5 = st.sidebar.checkbox("Different results")

    if optiune5:
        gruparea = cf['Country'].groupby(cf['Total Cases'] > 4000).count()
        st.write(gruparea)

        st.write(medie_total_teste())
        st.write(tara_totalcases())

        medie, disp, stdev, med, q = medie_dispersie_devstd_totalrec()
        st.subheader("Total Recovered analysis ")
        st.write("Mean: ", medie)
        st.write("Variance", disp)
        st.write("Standard Deviation: ", stdev)
        st.write("Mediane: ", med)
        st.write("Quartile ", q)

    X_, _ = modelarea()
    lsvariabile = []
    for i in X_.columns:
        lsvariabile.append(i)
    t1 = tuple(lsvariabile)
    st.subheader("Summary of the variables")
    sumar_variabile = st.selectbox("Select", t1)

    for variabila in X_.columns:
        if sumar_variabile == variabila:
            a, b, c, d, e = sumar_toate(variabila)
    st.write("Mean: ", a)
    st.write("Variance: ", b)
    st.write("Standard Deviation: ", c)
    st.write("Mediane: ", d)
    st.write("Quartile ", e)
    optiune6 = st.sidebar.checkbox("Relation Coefficients")
    if optiune6:
        # st.write("Correlation Coefficients between Total Cases and other variables, assumed as independent.")
        corelatie()

        # st.subheader("Relation Matrices")
        # matricea_heatmap()
        # matricea_heatmap_var_ind()

dataset_proc()

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
# name, authentication_status, username = authenticator.login('Login', 'main')
# if authentication_status:
#     authenticator.logout('Logout', 'sidebar')
#     st.sidebar.write(f'Welcome, *{st.session_state["name"]}*')
#     dataset_proc()
# else:
#     st.write("If you don't have an account, please register at the [link](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com/Registration)")


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')


st.markdown("<h2> Data Processing for the <i> Oceania Countries </i> Dataset (DS1) </h2> ", unsafe_allow_html=True)

# Citire si afisare date
df = pd.read_csv('oceania_covid.csv')
print(df.head(10).to_string(index=False))
df.head()
df.info()

# Tratare coloane nil
df.isnull().sum()
df = df.fillna(df.mean())
df.isnull().sum()
st.write(df.head())


# Vector de variabile independente si specificatie variabila dependenta
X = df.drop(['Country/Other', 'Total Cases'], axis=1)
y = df['Total Cases']

st.write(X)
st.write(y)


def corelatie():
    # print("\nVerif corelatia Pearson dintre variabile si Total Cases")
    st.write("\nCorelatia intre variabilele independente si cea dependenta ")
    for i in X.columns:
        corelatie, _ = pearsonr(X[i], y)
        st.write(i + ': %.2f' % corelatie)
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
st.write(medie_dispersie_devstd_Population())

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
    st.write("\nMatricea de relatie ")
    fig, ax = plt.subplots()
    mcorelatie = df.corr()
    sb.heatmap(mcorelatie, ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    # plt.show()
    st.write((fig))
# matricea_heatmap_total()

def matricea_heatmap_var_ind():
    st.write("Matricea de relatie pentru potentiala variabila independenta ")
    fig, ax = plt.subplots()
    mcorelatieX = X.corr()
    sb.heatmap(mcorelatieX, ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig("heatmap_ds1.pdf", format="pdf", bbox_inches="tight")
    # plt.show()
    st.write(fig)
matricea_heatmap_var_ind()


X = df[['Total Recovered', 'Active Cases', 'Total Tests', 'Tests/ 1M pop', 'Population']]
st.write(X)

