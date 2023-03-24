import streamlit as st


st.markdown('<h1 style="text-align: center;">  <br> </h1>', unsafe_allow_html=True)
st.markdown('<h4 style="text-align: center;"> Statistical indicators: Oceania <br> </h4>', unsafe_allow_html=True)

l = ["-----", "KNN", "SVM", "Decision Tree", "CART", "XGBoost"]
t = tuple(l)
coloanal = st.selectbox("Select the column ", t)

st.markdown('<h5 style="text-align: center;"> Mean squared error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/streamlit_img/knn_mse2.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/streamlit_img/svm_mse2.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/streamlit_img/dt_mse2.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/streamlit_img/cart_mse2.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/streamlit_img/xgb_mse2.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Mean absolute error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/streamlit_img/knn_mae2.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/streamlit_img/svm_mae2.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/streamlit_img/dt_mae2.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/streamlit_img/cart_mae2.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/streamlit_img/xgb_mae2.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Coefficient of determination <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/streamlit_img/knn_rs2.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/streamlit_img/svm_rs2.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/streamlit_img/dt_rs2.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/streamlit_img/cart_rs2.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/streamlit_img/xgb_rs2.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Correlation coefficient <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloanal == "KNN":
    imagine = "img/streamlit_img/knn_cp2.png"
    st.image(imagine)
elif coloanal == "SVM":
    imagine = "img/streamlit_img/svm_cp2.png"
    st.image(imagine)
elif coloanal == "Decision Tree":
    imagine = "img/streamlit_img/dt_cp2.png"
    st.image(imagine)
elif coloanal == "CART":
    imagine = "img/streamlit_img/cart_cp2.png"
    st.image(imagine)
elif coloanal == "XGBoost":
    imagine = "img/streamlit_img/xgb_cp2.png"
    st.image(imagine)



st.markdown('<hr> <br> <h4 style="text-align: center;"> Statistical indicators: Last 110 countries from Worldometer <br> </h4>', unsafe_allow_html=True)

l = ["-----", "KNN2", "SVM2", "Decision Tree2", "CART2", "XGBoost2"]
t = tuple(l)
coloana = st.selectbox("Select the column ", t)


st.markdown('<h5 style="text-align: center;"> Mean squared error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/streamlit_img/knn_mse1.png"
    st.image(imagine)
elif coloana == "SVM2":
    imagine = "img/streamlit_img/svm_mse1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = "img/streamlit_img/dt_mse1.png"
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/streamlit_img/cart_mse1.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/streamlit_img/xgb_mse1.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Mean absolute error <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/streamlit_img/knn_mae1.png"
    st.image(imagine)
elif coloana == "SVM2":
    imagine = "img/streamlit_img/svm_mae1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = "img/streamlit_img/dt_mae1.png"
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/streamlit_img/cart_mae1.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/streamlit_img/xgb_mae1.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Coefficient of determination <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/streamlit_img/knn_rs1.png"
    st.image(imagine)
elif coloana == "SVM2":
    imagine = "img/streamlit_img/svm_rs1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = "img/streamlit_img/dt_rs1.png"
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/streamlit_img/cart_rs1.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/streamlit_img/xgb_rs1.png"
    st.image(imagine)

st.markdown('<br> <h5 style="text-align: center;"> Correlation coefficient <br> </h5>', unsafe_allow_html=True)
result = "Please, select a dataset"
if coloana == "KNN2":
    imagine = "img/streamlit_img/knn_cp1.png"
    st.image(imagine)
elif coloana == "SVM":
    imagine = "img/streamlit_img/svm_cp1.png"
    st.image(imagine)
elif coloana == "Decision Tree2":
    imagine = "img/streamlit_img/dt_cp1.png"
    st.image(imagine)
elif coloana == "CART2":
    imagine = "img/streamlit_img/cart_cp1.png"
    st.image(imagine)
elif coloana == "XGBoost2":
    imagine = "img/streamlit_img/xgb_cp1.png"
    st.image(imagine)




