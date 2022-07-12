
import streamlit as st
import yaml
from streamlit_authenticator import Authenticate

st.set_page_config(
        page_title="ML Methods Analysis",
        page_icon="bar_chart",
        initial_sidebar_state="expanded"
    )

with open('../pages/config.yaml') as file:
    config = yaml.load(file, Loader=yaml.SafeLoader)


authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

try:
    if authenticator.register_user('Register user', preauthorization=False):
        st.success('User registered successfully')
    elif st.session_state["authentication_status"] == None:
        st.warning('Please enter your data')
except Exception as e:
    st.error(e)

with open('../pages/config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)

