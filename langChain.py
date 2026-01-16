import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Meu Assistente de Automação", page_icon="🤖")
st.title("💬 Chatbot de Automação (Gemini 2.5)")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("API_KEY"))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Pergunte sobre Playwright ou Cypress..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        resposta = llm.invoke(prompt)
        st.markdown(resposta.content)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta.content})