

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


def mainul():
    st.title("Prediction of COVID-19 using Machine Learning Models ")
    st.header("Model Evaluation Metrics and Statistical Indicators Computing ")

    st.subheader("Documentație")
    st.markdown('<div style="text-align: justify;"> În această lucrare se discută despre modul de implementare și modelele de învățare automată (ML) folosite în procesul de predicție \
         a datelor despre virusul COVID-19 care rezistă și după mai mult de doi ani. Scopul este de a se observa care model este \
         cel mai bun, cu cea mai mare acuratețe, sensibilitate, specificitate, precizie și scorul F1 și cel mai bun timp de execuție. \
         Se utilizează librăriile Scikit-learn și PySpark pentru implementare, pentru procesul de analiză statistică a datelor – indicând \
         coeficientul de determinare și coeficientul <i>Pearson</i>, la fiecare model și eroarea medie pătratică și cea absolută. \
         Algoritmii utilizați sunt Regresia liniară (LR) utilizate la partea de construire a unei predicții conform unui model liniar, \
         apoi <i>K-cei mai apropiați vecini</i> (KNN), <i>Mașina de suport vectorială</i> (SVM), <i>Arborele de decizie</i> (DT), \
         <i>Arborii de clasificare și regresie</i> (CART) și <i>XGBoost</i>. '
                '<br>&emsp; Setul de date selectat ține de datele de bază despre ultimele 100 de țări de pe \
         Worldometer până pe septembrie 2020. Se va face o predicție în conformitate cu variabila răspuns, care este <i>Total Cases</i>, \
         pentru o anumită țară. \
        În acest set de date sunt cuprinse toate datele despre pacienți, sau mai precis populație, din decembrie 2019, \
        când a fost înregistrat primul caz de Sars-CoV-2, și până pe septembrie din anul 2020. Deoarece la această etapă \
        a proiectului se discută despre metodele de culegere a datelor și despre importanța acestora, este important de \
        precizat căci designul acestui studiu este prospectiv, fiindcă se sprijină pe datele deja culese, însă se dorește o \
        predicție, o prognoză viitoare a unui parametru, acesta fiind în cazul de față Total Cases, care este variabila dependentă \
        la procesul de instruire și testare a sistemului. \
        <br>&emsp; Setul de date a fost preluat de pe <i>Kaggle</i>. S-au căutat date despre un set de intrare care să se potrivească cu o cerință \
        specificată de creatorii competiției de <i>Covid Forecasting</i> de pe Kaggle, în anul 2021. Setul de date a fost găsit după ce pe \
        Kaggle s-a căutat un set de date corelat cu cele specificate sau indicate în cerințele problematici de pe Kaggle. Pentru această \
        competiție s-a cerut să se găsească un set de date nu atât de mare și să se găsească cel mai performant model de învățare automată. \
        Acest experiment se referă la implementarea a câteva metode de învățare automată, cu tendința de a se ajunge la cea mai bună proporție \
        în ceea ce privește diferite indicatoare de performanță, cum ar fi timpul de execuție, coeficientul de determinare (scorul R-squared), \
        eroarea medie pătratică (MSE), acuratețea, sensibilitatea, specificitatea și altele. \
        <br>&emsp; O problemă de etică din cadrul acestui set de date este faptul că persoanele testate nu și-au exprimat învoiala ca datele lor să \
        fie introduse în sistem. O altă problemă de etică, care totuși ține de epidemiologia genetică, este utilizarea diferitelor aparate \
        tehnologice în astfel de studii. O altă problematică de etică se leagă cumva de primul exemplu indicat; pacienții nu-și oferă \
        consințământul, nu le sunt indicate beneficiile și dezavantajele de participare la construirea unui set de date, care ulterior va \
        fi folosit pentru a construi diverse experimente.  </div>', unsafe_allow_html=True)


footer()
mainul()
