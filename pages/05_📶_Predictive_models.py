import streamlit as st


st.markdown('<h1 style="text-align: center;">  <br> </h1>', unsafe_allow_html=True)
st.markdown('<h4 style="text-align: center;"> Statistical indicators: Oceania <br> </h4>', unsafe_allow_html=True)

l = ["-----", "KNN", "SVM", "Decision Tree", "CART", "XGBoost"]
t = tuple(l)
coloanal = st.selectbox("Select the column ", t)

st.markdown('<h5 style="text-align: center;"> Mean squared error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/alte/knn_mse_ds1.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/alte/svm_mse_ds1.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/alte/dt_mse_ds1.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/alte/cart_mse_ds1.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/alte/xgboost_mse_ds1.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Mean absolute error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/alte/knn_mae_ds1.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/alte/svm_mae_ds1.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/alte/dt_mae_ds1.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/alte/cart_mae_ds1.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/alte/xgboost_mae_ds1.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Coefficient of determination <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/alte/knn_coefdet_ds1.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/alte/svm_coefdet_ds1.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/alte/dt_coefdet_ds1.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/alte/cart_coefdet_ds1.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/alte/xgboost_coefdet_ds1.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Correlation coefficient <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/alte/knn_coefcp_ds1.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/alte/svm_coefcp_ds1.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/alte/dt_coefcp_ds1.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/alte/cart_coefcp_ds1.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/alte/xgboost_coefcp_ds1.png"
    st.image(imagine)



st.markdown('<hr> <br> <h4 style="text-align: center;"> Statistical indicators: Last 100 countries from Worldometer (DS2) <br> </h4>', unsafe_allow_html=True)

l = ["-----", "KNN2", "SVM2", "DT2", "CART2", "XGBoost2"]
t = tuple(l)
coloana = st.selectbox("Select the column ", t)


st.markdown('<h5 style="text-align: center;"> Mean squared error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/alte/knn_mse_ds2.png"
    st.image(imagine)
elif coloana == "SVM2":
    imagine = "img/streamlit_img/svm_mse1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = ""
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/alte/cart_mse_ds2.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/alte/xgboost_mse_ds2.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Mean absolute error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/alte/knn_mae_ds1.png"
    st.image(imagine)
elif coloana == "SVM2":
    imagine = "img/streamlit_img/svm_mae1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = ""
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/alte/cart_mae_ds2.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/alte/xgboost_mae_ds2.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Coefficient of determination <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/alte/knn_coefdet_ds2.png"
    st.image(imagine)
elif coloana == "SVM2":
    imagine = "img/alte/svm_coefdet_ds1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = ""
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/alte/cart_coefdet_ds2.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/alte/xgboost_coefdet_ds2.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Correlation coefficient <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/alte/knn_coefcp_ds2.png"
    st.image(imagine)
elif coloana == "SVM":
    imagine = "img/alte/svm_coefcp_ds1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = ""
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/alte/cart_coefcp_ds2.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/alte/xgboost_coefcp_ds2.png"
    st.image(imagine)




