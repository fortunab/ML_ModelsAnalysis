

import streamlit as st
# import numpy
# import yaml
# from streamlit_authenticator import Authenticate
#
# from footerul import footer
# from grafice_basic_pandas import afisarea_grafice_concret, afisarea_grafice_general
# from input_pandas import citire_file
# from model_liniar_alg_pandas import output_model
# from modele_ML import plotare
# from prelucrare_date_pandas import medie_modif, medie_total_teste, tara_totalcases, corelatie, \
#     matricea_heatmap, medie_dispersie_devstd_totalrec, sumar_toate, modelarea, afisarea_prelucrarea
#
# from streamlit_option_menu import option_menu

from footerul import footer

st.set_page_config(
        page_title="ML Methods Analysis",
        page_icon="🌁",
    )


def mainul():
    st.title("Machine Learning Predictive Models on COVID-19 Datasets to "
             "Estimate the Infection Risk in Certain Geographic Regions")
    st.header("Models Evaluation Metrics and Statistical Indicators Calculation ")


    st.markdown('<div style="text-align: justify;"> Hello! Bienvenue to the application. '
                '<br> 👈 Select a page from the sidebar to see some examples. </div>', unsafe_allow_html=True)

footer()
mainul()
