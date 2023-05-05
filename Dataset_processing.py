
import matplotlib.pyplot as plt
import numpy
import seaborn as sb
import streamlit as st
import yaml
from scipy.stats import pearsonr
from streamlit_authenticator import Authenticate

from footerul import footer
from input_pandas import citire_file



# st.set_page_config(
#         page_title="ML Methods Analysis",
#         page_icon="bar_chart",
#         initial_sidebar_state="expanded"
#     )

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
    travg = cf["Total Recovered"].mean()
    # dispersie coloana Total Recovered
    trv = cf["Total Recovered"].var()
    # deviatie standard coloana Total Recovered
    trsd = cf["Total Recovered"].std()
    # mediana coloana Total Recovered
    trmed = cf["Total Recovered"].median()
    # cuartila coloana Total Recovered
    trq = cf["Total Recovered"].quantile([0.25, 0.5, 0.75])
    return travg, trv, trsd, trmed, trq

def sumar_toate(col):
    cf = medie_modif()
    # medie coloana
    avg = cf[col].mean()
    # dispersie coloana
    v = cf[col].var()
    # deviatie standard coloana
    sd = cf[col].std()
    # mediana coloana
    med = cf[col].median()
    # cuartile coloana
    q = cf[col].quantile([0.25, 0.5, 0.75])
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

    st.header("Data Processing ")

    # nume_ds = st.selectbox("Select dataset", ("-----", "OfficialSeptember2020"))
    # if nume_ds == "OfficialSeptember2020":
    #     st.write(nume_ds + "DateSeptembrie2020_augmentat.csv")

    footer()
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
        st.write("Total number of countries recognised as independent from the list: ", len(numpy.unique(y)))

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
    sumar_variabile = st.selectbox("Summary of the variables", t1)

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

