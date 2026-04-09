import streamlit as st

st.set_page_config(page_title="ForestWatch", page_icon="🌿", layout="wide")

pg = st.navigation([
    st.Page("pages/intro.py",    title="🌍 Why This Project",   default=True),
    st.Page("pages/preprocess.py", title="🔧 Data & Preprocessing"),
    st.Page("pages/eda.py",      title="📊 EDA"),
    st.Page("pages/model.py",    title="🤖 Clustering & Results"),
])
pg.run()