

import streamlit as st
import numpy
import yaml
from streamlit_authenticator import Authenticate

from footerul import footer
from grafice_basic_pandas import afisarea_grafice_concret, afisarea_grafice_general
from input_pandas import citire_file
from model_liniar_alg_pandas import output_model
from modele_ML import plotare
from prelucrare_date_pandas import medie_modif, medie_total_teste, tara_totalcases, corelatie, \
    matricea_heatmap, medie_dispersie_devstd_totalrec, sumar_toate, modelarea, afisarea_prelucrarea

from streamlit_option_menu import option_menu


st.set_page_config(
        page_title="ML Methods Analysis",
        page_icon="bar_chart",
        initial_sidebar_state="expanded"
    )

with st.sidebar:
    choose = option_menu("Pages",
            ["Home Tours", "Data Processing", "ML Models Analysis"],
            icons=['house', 'palette2', 'kanban'],
            # 'bar-chart-steps', 'funnel'
            menu_icon="app-indicator", default_index=0,
            styles={
                    "container": {"padding": "5!important", "background-color": "#fafafa"},
                    "icon": {"color": "orange", "font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                    "nav-link-selected": {"background-color": "#02ab21"},
            }
    )


def mainul():
    st.title("Prediction of COVID-19 using Machine Learning Models ")
    st.header("Model Evaluation Metrics and Statistical Indicators Computing ")

    # choose = st.sidebar.radio("Pages",
    #                           ("Home Tours", "Data Processing", "ML Models Analysis"))

    # nume_class = st.selectbox("Alege clasificatorul", ("Linear Regression", "KNN", "Decision Tree", "SVM"))

    # sidebarul = st.sidebar.radio("Pagini", ("Prelucrare date", "Afisari grafice de baza", "Model liniar cu Linear Regression", "Comparatie algoritmi invatare automata"))

    if choose == "Data Processing":
        footer()
        optiune1 = st.sidebar.checkbox("Initial dataset")
        optiune2 = st.sidebar.checkbox("Dataset with valid values")

        ds = citire_file()
        if optiune1:
            st.write(ds)

        cf = medie_modif()
        if optiune2:
            st.write(cf)

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
            afisarea_prelucrarea()
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
            st.write("Quantile: ", q)

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
        st.write("Quantile: ", e)
        optiune6 = st.sidebar.checkbox("Relation Coefficients")
        if optiune6:
            # st.write("Correlation Coefficients between Total Cases and other variables, assumed as independent.")
            corelatie()
            # st.subheader("Relation Matrices")
            # matricea_heatmap()
            # matricea_heatmap_var_ind()


    if choose == "Basic Graphs":
        footer()
        afisarea_grafice_concret()
        X_, _ = modelarea()
        l = []
        for i in X_.columns:
            l.append(i)
        t = tuple(l)
        coloana = st.selectbox("Select the column ", t)

        for i in X_.columns:
            if coloana==i:
                afisarea_grafice_general(col=i)

    if choose == "Regression Models":
        footer()

        f_optiu = st.sidebar.checkbox("Relation matrix")
        if f_optiu:
            matricea_heatmap()
            # matricea_heatmap_var_ind()
        output_model()

    nume_ds = st.selectbox("Select dataset", ("-----", "OfficialSeptember2020", "Oceania"))

    result = "Please, select a dataset"
    if nume_ds == "OfficialSeptember2020":
        result = "septembrie2020_augmentat.csv"
    elif nume_ds == "Oceania":
        result = "oceania_covid.csv"
    st.write(result)

    if choose == "ML Models Analysis":
        # plotare()
        if nume_ds == "OfficialSeptember2020":
            imagine = "img/exacta_septembrie2020_augmentat_solutie_5_metode_alta.png"
            st.image(imagine)
        if nume_ds == "Oceania":
            imagine = "img/metode_5_solutie_oceania_dataset.png"
            st.image(imagine)

footer()
mainul()

# with open('pages/config.yaml') as file:
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
#     mainul()
# else:
#     st.write("If you don't have an account, please register at the [link](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com/Registration)")
