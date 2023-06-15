"""
def modelul():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_) # fitting, potrivirea de date
    return model

def model_LR_coeffs():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_)
    r_sq = model.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r  # coeficientul de determinare si coeficientul de relatie

def model_LR_msq_mabs_e():
    model = LinearRegression()
    X_, y_ = modelarea()
    model = model.fit(X_, y_)
    y_pred = model.predict(X_)
    mse = mean_squared_error(y_, y_pred)
    mabs = mean_absolute_error(y_, y_pred)
    return mse, mabs

def model_DT_coeffs():
    dc = tree.DecisionTreeClassifier(criterion="entropy", max_depth=2)
    X_, y_ = modelarea()
    dc = dc.fit(X_, y_)
    r_sq = dc.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_DT_msq_mabs_e():
    model = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    mse = mean_squared_error(y_pred, y_test.values)
    mabs = mean_absolute_error(y_pred, y_test.values)
    return mse, mabs

def model_SVM_coeffs():
    modelsvc = SVC(C=0.5, kernel="poly", degree=3, decision_function_shape="ovo")
    X_, y_ = modelarea()
    modelsvc = modelsvc.fit(X_, y_)
    r_sq = modelsvc.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_SVM_msq_mabs_e():
    model = SVC(C=0.5, kernel="poly", degree=5, decision_function_shape="ovo")
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    mse = mean_squared_error(y_pred, y_test.values)
    mabs = mean_absolute_error(y_pred, y_test.values)
    return mse, mabs

def model_KNN_coeffs():
    modelknn = KNeighborsClassifier(n_neighbors=10)
    X_, y_ = modelarea()
    modelknn = modelknn.fit(X_, y_)
    r_sq = modelknn.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_KNN_msq_mabs_e():
    model = KNeighborsClassifier(n_neighbors=10)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    mse = mean_squared_error(y_pred, y_test.values)
    mabs = mean_absolute_error(y_pred, y_test.values)
    return mse, mabs

def model_CART_coeffs():
    cart = tree.DecisionTreeClassifier(criterion="gini", max_depth=25)
    X_, y_ = modelarea()
    cart = cart.fit(X_, y_)
    r_sq = cart.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_CART_msq_mabs_e():
    model = tree.DecisionTreeClassifier(criterion="gini", max_depth=25)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)
    model = model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    mse = mean_squared_error(y_pred, y_test.values)
    mabs = mean_absolute_error(y_pred, y_test.values)
    return mse, mabs

def model_XGBoost_coeffs():
    # X_train.values, X_test.values, y_train.values, y_test.values = train_test_split(X, y, test_size=0.3, random_state=False)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # y_train.values = le.fit_transform(y_train.values)

    # xboost = XGBClassifier(subsample=0.15, max_depth=2)
    # rezultat = cross_val_score(xboost, X_train.values, y_train.values, cv=5, scoring='accuracy')
    xgb = XGBClassifier(subsample=0.15, max_depth=2)
    X_, y_ = modelarea()
    y_ = le.fit_transform(y_)
    xgb = xgb.fit(X_, y_)
    r_sq = xgb.score(X_, y_)
    r = sqrt(r_sq)
    return r_sq, r

def model_XGBoost_msq_mabs_e():
    model = XGBClassifier(subsample=0.15, max_depth=2)
    X_, y_ = modelarea()
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.3, random_state=False)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # sau fara values
    y_train.values = le.fit_transform(y_train.values)

    model = model.fit(X_train.values, y_train.values)
    y_pred = model.predict(X_test.values)
    mse = mean_squared_error(y_pred, y_test.values)
    mabs = mean_absolute_error(y_pred, y_test.values)
    return mse, mabs

def predictie_concret(real):
    model = modelul()
    y_tara_tc = model.predict(real)
    return y_tara_tc

def predictie_general():
    model = modelul()
    X_, _ = modelarea()
    y_pred = model.predict(X_)
    df = pd.DataFrame(y_pred, columns=['Prediction value'])
    return df

def output_model():
    lr_r_sq, lr_r = model_LR_coeffs()
    st.subheader("Linear Regression Algorithm Example")
    st.write("Determination coefficient: ", lr_r_sq)
    st.write("Relation coefficient: ", lr_r)

    uk = numpy.array([68592949.0, 522526476.0, 22142505.0, 156.0, 348910.0]).reshape((1, -1))
    y_uk = predictie_concret(uk)
    st.write('Prediction of Total Cases in United Kingdom: ', round(y_uk[-1]))



def timpii_executie():
    start_time = time.time()
    model_LR_coeffs()
    linr = "Linear Regression --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_KNN_coeffs()
    knn = "KNN --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_SVM_coeffs()
    polysvm = "SVM --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_DT_coeffs()
    dc = "Decision Tree Model --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_CART_coeffs()
    cart = "CART --- %s seconds ---" % (time.time() - start_time)

    start_time = time.time()
    model_XGBoost_coeffs()
    xgb = "XGB --- %s seconds ---" % (time.time() - start_time)


    return linr, knn, polysvm, dc, cart, xgb

# a, b, c, d, e, f = timpii_executie()
# for i in timpii_executie():
#     st.write(i)

def grafic_timpii_executie():
    start_time = time.time()
    model_LR_coeffs()
    linr = time.time() - start_time

    start_time = time.time()
    model_KNN_coeffs()
    knn = time.time() - start_time

    start_time = time.time()
    model_SVM_coeffs()
    polysvm = time.time() - start_time

    start_time = time.time()
    model_DT_coeffs()
    dc = time.time() - start_time

    start_time = time.time()
    model_CART_coeffs()
    cart = time.time() - start_time

    start_time = time.time()
    model_XGBoost_coeffs()
    xgb = time.time() - start_time


    return linr, knn, polysvm, dc, cart, xgb


a, b, c, d, e, f = grafic_timpii_executie()

def grafic_te():
    data = {"Linear Regression":a, "KNN":b, "SVM":c, "Decision Tree Model":d, "CART":e, "XGBoost":f}
    modele = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(modele, values, color='blue', width=0.4)

    plt.xlabel("Models")
    plt.ylabel("Seconds")
    plt.title("Execution time ")

    plt.savefig("timp_executieDS2.pdf", format="pdf", bbox_inches="tight")
    st.pyplot(fig=plt)
# grafic_te()
te = st.checkbox("Execution time ")

if te:
    # grafic_te()
    imaginea = "img/timpul_executie_modele.png"

    st.image(imaginea)


f_optiu = st.sidebar.checkbox("Relation Matrices ")
if f_optiu:
    matricea_heatmap()
    matricea_heatmap_var_ind()
output_model()
"""

"""
feedback = st.checkbox("Feedback")
if feedback:
    with st.form("Feedback"):
            st.header("Feedback")
            val = st.selectbox("How was your experience of this application?", ["-----", "Good", "Neutral", "Bad"])
            st.select_slider("How would you rate the application",
                             ["Poor", "Not Good", "As Expected", "Easy for follow", "Excellent"], value="As Expected")
            st.form_submit_button("Submit")
            if val != "-----":
                st.text("Thank you for your implication and for the feedback ")
"""


# with open('config.yaml') as file:
#     config = yaml.load(file, Loader=yaml.SafeLoader)
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
#     modele()
# else:
#     st.write("If you don't have an account, please register at the [link](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com/Registration)")


# sensibilitatea()
# specificitatea()
# precizia()
# scorulF1()

chimg1 = st.checkbox("Activate Sensitivity and Specificity results")
if chimg1:
    imgsensibilitate = "img/sensibilitatea.png"
    st.image(imgsensibilitate)
    imgspec = "img/specificitatea.png"
    st.image(imgspec)

chimgalt = st.checkbox("Activate Precision and F1 Score results")
if chimgalt:
    imgprec = "img/precizia.png"
    st.image(imgprec)
    imgf1 = "img/scorulF1.png"
    st.image(imgf1)


