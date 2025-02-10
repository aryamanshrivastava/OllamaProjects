import streamlit as st
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = Ollama(model = "llama3.2:1b",base_url = "http://localhost:11434")

def sendPrompt(prompt):
    global llm
    response = llm.invoke(prompt)
    return response

st.title("ChatBot")
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role":"assistant","content":"Ask me a question !"}
    ]

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role":"user","content":prompt})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"]!="assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = sendPrompt(prompt)
            print(response)
            st.write(response)
            message = {"role":"assistant","content":response}
            st.session_state.messages.append(message)