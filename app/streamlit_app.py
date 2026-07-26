import requests
import streamlit as st

st.set_page_config(page_title="Support Assistant", page_icon="🛠️")
st.title("🛠️ Instruction-Tuned Support Assistant")
url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
temp = st.sidebar.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
q = st.text_area(
    "Question",
    "My Docker container exits immediately after startup. How do I troubleshoot it?",
)
if st.button("Generate", type="primary"):
    try:
        r = requests.post(
            url.rstrip("/") + "/generate",
            json={"question": q, "temperature": temp},
            timeout=300,
        )
        r.raise_for_status()
        st.markdown(r.json()["answer"])
    except requests.RequestException as e:
        st.error(str(e))
