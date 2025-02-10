import streamlit as st
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

def load_excel_data(file_path):
    try:
        df = pd.read_excel(file_path)
        return "\n".join([str(row) for row in df.to_dict("records")])
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return ""

def setup_chain():
    llm = OllamaLLM(model="llama3.2:1b", temperature=0.1)

    prompt = ChatPromptTemplate.from_template(
        """You are an AI assistant specializing in structured data analysis. Answer STRICTLY based on the given dataset.
        ### **Dataset Context:**
        {context}
        ### **Rules for Response:**
        Provide ONLY exact values from the data. **No explanations.**  
        If data is missing, reply with: **"Invalid information."**  
        **No greetings, acknowledgments, or extra words.**  
        When multiple values exist, return them as a **bullet list.**  
        Do NOT format responses with Markdown, code, or tables.  
        **Question:** {query}  
        **Answer:** """
    )

    return prompt | llm | StrOutputParser()

def process_excel(file_path, user_query=""):
    text_data = load_excel_data(file_path)
    
    if not text_data:
        return "⚠️ No data available for processing."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text_data)
    
    chain = setup_chain()
    insights = []
    
    for chunk in chunks:
        try:
            response = chain.invoke({"context": chunk, "query": user_query})
            insights.append(response)
        except Exception as e:
           return f"❌ Error processing data: {e}"
    return "\n".join(insights)

# Streamlit UI Configuration
st.set_page_config(page_title="Excel Chatbot", page_icon="📊", layout="wide")

# Sidebar for file upload
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("📂 Upload an Excel File", type=["xlsx", "xls"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

st.title("📊 Excel Chatbot")
st.write("Upload an **Excel file** and ask data-related questions.")

if st.session_state.uploaded_file:
    
    df = pd.read_excel(st.session_state.uploaded_file)
    
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(10))

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if user_query := st.chat_input("Ask a question about your Excel data..."):
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # Process query
        with st.spinner("Analyzing..."):
            response = process_excel(st.session_state.uploaded_file, user_query)
        
        # Display AI response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
else:
     st.warning("⚠️ Please upload an Excel file to start chatting.")

# Always show the chat input (disabled until file upload)
if not st.session_state.uploaded_file:
    st.chat_input("Upload a file to begin chatting...", disabled=True)
