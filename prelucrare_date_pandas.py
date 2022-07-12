
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st


from input_pandas import citire_file


def medie_modif():
    cf = citire_file()
    # se afiseaza media tuturor coloanelor numerice
    # valorile NaN se inlocuiesc cu media valorilor
    # de pe coloana respectiva, pentru fiecare coloana
    ult = cf.fillna(cf.mean())
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
    # cuantila coloana Total Recovered
    trq = cf["Total Recovered"].quantile()
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
    # cuantila coloana
    q = cf[col].quantile()
    return avg, v, sd, med, q

def noua_col():

    cf = citire_file()
    recov = cf["Total Recovered"]
    total = cf["Total Cases"]
    l_proc = []
    for i in range(len(recov)):
        # convertire in integer deoarece la completarea
        # campurilor vide s-a adaugat o valoare reala
        # iar numarul de zecimale nu se potriveste cu
        # numarul obtinut la media valorilor
        # de pe acea coloana
        if int(recov[i]) == int(recov.mean()):
            l_proc.append(None)
        else:
            proc_recov_pe_total = recov[i] / total[i]
            l_proc.append(proc_recov_pe_total)
    print(l_proc)
    st.write("Mean value of the recovered ones: ", int(recov.mean()))

    # se adauga noua coloana
    cf["Total Recovered/ Total Cases"] = l_proc

    # se completeaza campurile nan cu media de pe acea coloana
    cf["Total Recovered/ Total Cases"].fillna(cf["Total Recovered/ Total Cases"].mean())

    tr_tc = cf["Total Recovered/ Total Cases"].max()

    for indice, rand in cf.iterrows():

        if rand["Total Recovered/ Total Cases"] == tr_tc:
            st.write(str(rand["Country"]) +
                  " had the biggest ratio of Total Recovered per Total Cases "
                  "and its value is: " + str(tr_tc))
            break

    active = cf["Active Cases"].max()
    for indice, rand in cf.iterrows():
        if rand["Active Cases"] == active:
            st.write(str(rand["Country"]) +
                  " had the most active cases, precisely it is: " +
                  str(active))
            break


# Modelare date: X = predictor, var independenta, y = raspuns, var dependenta

# Pregatire date pentru modelul liniar
def modelarea():
    cf = medie_modif()
    X = cf.drop(columns=["Country", "Total Cases", "Total Deaths"])
    y = cf["Total Cases"]
    return X, y

def corelatie():
    # print("\nVerif corelatia Pearson dintre variabile si Total Cases")
    X, y = modelarea()
    st.subheader("\nVerification of Pearson correlation between the independent variables and Total Cases")
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

# def matricea_heatmap_var_ind():
#     print("Matricea de corelatie")
#     X_, _ = new_modelarea()
#     fig, ax = plt.subplots()
#     mcorelatieX = X_.corr()
#     sb.heatmap(mcorelatieX, ax=ax)
#     plt.xticks(rotation=45)
#     plt.yticks(rotation=45)
#     # plt.show()
#     st.write(fig)


def afisarea_prelucrarea():
    cf = medie_modif()
    st.write("Number of observations with a Total Cases value bigger or less than 4k, for the dataset: ", cf["Country"].groupby(cf['Total Cases']>4000).count())
    st.write(medie_total_teste())
    st.write("Country | Total Cases: ", tara_totalcases().head(10))
    X, y = modelarea()
    X.head()
    y.head()
    noua_col()

    corelatie()
    matricea_heatmap()
    # matricea_heatmap_var_ind()


# afisarea_prelucrarea()

