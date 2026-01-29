import streamlit as st
import os
import tempfile
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- Page Config ---
st.set_page_config(page_title="Ken Lam's AI RAG Agent", layout="wide", page_icon="🤖")

# --- UI Header & Personal Branding ---
st.title("🤖 Python AI RAG Agent Demo by Ken Lam")
st.subheader("Interactive Product Knowledge Retrieval System")

# --- Sidebar: Profile & Config ---
with st.sidebar:
    # --- Personal Branding Section ---
    st.header("👨‍💻 Developed By")
    st.markdown("""
    **Ken Lam** [🔗 LinkedIn](https://www.linkedin.com/in/ken-yiu-kei-lam/) [📧 Email](mailto:kenyklam330@gmail.com)
    """)
    #st.divider()

    # --- Model Configuration ---
    st.header("⚙️ Model Settings")
    provider = st.selectbox("Select Provider", ["OpenAI (GPT-4o-mini) [API KEY REQUIRED]", "Ollama (Gemma 3) [FOR LOCAL TESTING ONLY]"])
    
    if provider == "OpenAI (GPT-4o-mini) [API KEY REQUIRED]":
        api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    else:
        api_key = None
        st.error("Ensure Ollama is running locally with Gemma 3 & Nomic-Embed-Text.")

    uploaded_file = st.file_uploader("📂 Upload Product PDF", type="pdf")
    
    if st.button("🗑️ Reset Chat"):
        st.session_state.clear()
        st.rerun()

# --- Model & Embedding Factory ---
def get_model_and_embeddings(provider, key=None):
    if provider == "OpenAI (GPT-4o-mini)":
        if not key:
            st.error("Please enter an OpenAI API Key.")
            st.stop()
        llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=key)
        embeddings = OpenAIEmbeddings(openai_api_key=key)
    else:
        # Gemma 3:4b Brain + Nomic Embeddings Librarian
        llm = ChatOllama(model="gemma3:4b", temperature=0, num_ctx=8000)
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return llm, embeddings

# --- RAG Logic ---
def process_pdf(file, embeddings):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(splits, embeddings)
    os.remove(tmp_path)
    return vectorstore.as_retriever()

# --- Main App Logic ---
if uploaded_file:
    llm, embeddings = get_model_and_embeddings(provider, api_key)
    
    if "retriever" not in st.session_state or st.session_state.get("last_provider") != provider:
        with st.spinner(f"Indexing PDF with {provider}..."):
            st.session_state.retriever = process_pdf(uploaded_file, embeddings)
            st.session_state.last_provider = provider

    # Setup RAG Chain
    prompt = ChatPromptTemplate.from_template("Using the context: {context}\n\nAnswer the question: {input}")
    combine_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(st.session_state.retriever, combine_chain)

    # Chat Memory UI
    if "messages" not in st.session_state: st.session_state.messages = []
    for msg in st.session_state.messages: st.chat_message(msg["role"]).write(msg["content"])

    if query := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                res = rag_chain.invoke({"input": query})
                st.write(res["answer"])
                st.session_state.messages.append({"role": "assistant", "content": res["answer"]})
else:
    st.write("---")
    st.info("👈 Please upload a document in the sidebar to start.")