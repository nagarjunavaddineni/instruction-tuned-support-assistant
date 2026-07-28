import requests
import streamlit as st

st.set_page_config(page_title="Support Assistant", page_icon="🛠️")
st.title("🛠️ Instruction-Tuned Support Assistant")

url = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
temp = st.sidebar.slider("Temperature", 0.0, 1.5, 0.3, 0.1)
if st.sidebar.button("Clear conversation"):
    st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "My Docker container exits immediately after startup. How do I troubleshoot it?"
)
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            r = requests.post(
                url.rstrip("/") + "/generate",
                json={
                    "question": question,
                    "temperature": temp,
                    "history": st.session_state.messages[:-1][-20:],
                },
                timeout=300,
            )
            r.raise_for_status()
            answer = r.json()["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except requests.RequestException as e:
            st.error(str(e))
