
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from streamlit_authenticator import Authenticate
from Application import mainul

st.set_page_config(
        page_title="ML Methods Analysis",
        page_icon="bar_chart",
        initial_sidebar_state="expanded"
    )


def logarea():
    hashed_passwords = stauth.Hasher(['123', '456']).generate()

    with open('../pages/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)

    authenticator = Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )



    name, authentication_status, username = authenticator.login('Login', 'main')
    if authentication_status:
        st.write(f'Congratulations *{st.session_state["name"]}*, you are now logged in')
        st.write("Click [here](https://fortunab-ml-methods-application-kqtq2p.streamlitapp.com) to continue")
    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')

    #
    # if authentication_status:
    #     try:
    #         if authenticator.reset_password(username, 'Reset password'):
    #             st.success('Password modified successfully')
    #     except Exception as e:
    #         st.error(e)

    # try:
    #     username_forgot_pw, email_forgot_password, random_password = authenticator.forgot_password('Forgot password')
    #     if username_forgot_pw:
    #         st.success('New password sent securely')
    #         # Random password to be transferred to user securely
    #     elif username_forgot_pw == False:
    #         st.error('Username not found')
    # except Exception as e:
    #     st.error(e)

    # try:
    #     username_forgot_username, email_forgot_username = authenticator.forgot_username('Forgot username')
    #     if username_forgot_username:
    #         st.success('Username sent securely')
    #         # Username to be transferred to user securely
    #     elif username_forgot_username == False:
    #         st.error('Email not found')
    # except Exception as e:
    #     st.error(e)

    # if authentication_status:
    #     try:
    #         if authenticator.update_user_details(username, 'Update user details'):
    #             st.success('Entries updated successfully')
    #     except Exception as e:
    #         st.error(e)

    with open('../pages/config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


logarea()


