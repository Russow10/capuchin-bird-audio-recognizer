# pages/3_Chat_Assistant.py
import streamlit as st
import time
import os
from dotenv import load_dotenv
import gc
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Audio Assistant", page_icon="💬")
st.title("Audio Analysis Assistant")
st.sidebar.header("Chat Options")

# Initialize session state variables if they don't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'loading_status' not in st.session_state:
    st.session_state.loading_status = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Check for API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY not found in environment variables. Please set it up.")

def load_documents():
    with st.spinner("Loading knowledge base..."):
        st.session_state.loading_status = "Loading documents"
        knowledge_dir = "docs"
        documents = []
        
        if os.path.exists(knowledge_dir):
            for file in os.listdir(knowledge_dir):
                file_path = os.path.join(knowledge_dir, file)
                if file.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
                elif file.lower().endswith('.txt'):
                    loader = TextLoader(file_path)
                    loaded_docs = loader.load()
                    documents.extend(loaded_docs)
        else:
            st.error(f"Knowledge directory '{knowledge_dir}' not found. Please ensure it exists with PDF files.")
            return False

        st.session_state.loading_status = "Splitting text"
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        st.session_state.loading_status = "Creating embeddings"
        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory="./temp_chroma_db"
        )
        st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        
        st.session_state.loading_status = "Setting up RAG chain"
        llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192", api_key=api_key)
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert audio analysis assistant that helps users understand their audio files and analysis results.
        
        Answer the user's question based on the following retrieved context:
        
        {context}
        
        If the context is insufficient, supplement your answer with up-to-date internet knowledge.
        Provide detailed and clear explanations.
        
        User question: {input}
        """)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        st.session_state.rag_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)
        
        st.session_state.documents_loaded = True
        st.session_state.loading_status = None
        gc.collect()
        return True

def generate_response(query):
    if not st.session_state.documents_loaded:
        if not load_documents():
            return "I'm having trouble loading my knowledge base. Please ensure your PDF files are in the docs folder."
    try:
        audio_context = ""
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            audio_context = f"""
            The user has analyzed an audio file with the following properties:
            - Filename: {results['filename']}
            - Duration: {results['duration']:.2f} seconds
            - Sample Rate: {results['sample_rate']} Hz
            """
            if 'tempo' in results and results['tempo'] > 0:
                audio_context += f"- Estimated Tempo: {results['tempo']:.2f} BPM\n"
            if 'capuchin_calls' in results:
                audio_context += f"- Capuchin Call Count: {results['capuchin_calls']}\n"
        enhanced_query = f"{query}\n\nAudio file context: {audio_context}" if audio_context else query
        
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke({"input": enhanced_query})
        answer = response.get("answer", "I'm sorry, I could not generate an answer.")
        return answer
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while processing your question. Please try again. Details: {str(e)}"

# Sidebar actions
if st.sidebar.button("Load Knowledge Base") and not st.session_state.documents_loaded:
    load_documents()

if st.sidebar.button("Reload Knowledge Base"):
    st.session_state.documents_loaded = False
    load_documents()

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.experimental_rerun()

if st.session_state.loading_status:
    st.info(f"Status: {st.session_state.loading_status}")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input area
user_input = st.chat_input("Ask about audio analysis...")
if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = generate_response(user_input)
        placeholder.markdown(full_response)
    
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
