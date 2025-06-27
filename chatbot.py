import streamlit as st
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from config import DATA_DIRECTORY, COLLECTION_NAME, set_environment
from dotenv import load_dotenv # To load environment variables from .env file

# Load environment variables from .env file
load_dotenv()
from dotenv import load_dotenv

# --- Configuration (Azure OpenAI & Data Paths) ---
PERSIST_DIRECTORY = "./chroma_db"  # Directory to store Chroma DB data persistently
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_LLM_API_VERSION = os.getenv("AZURE_OPENAI_API_LLM_VERSION", "2024-02-15-preview") # Default if not set
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT") # e.g., "gpt-35-turbo-16k"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") # e.g., "text-embedding-ada-002"
AZURE_OPENAI_API_EMBEDDING_VERSION = os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION", "2024-02-15-preview") # Default if not set




# --- Validate Configuration ---
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_CHAT_DEPLOYMENT, AZURE_OPENAI_LLM_API_VERSION]):
    st.error("Azure OpenAI configuration missing. Please set environment variables (or Streamlit secrets for deployment).")
    st.stop()

st.title("Local Document RAG Chatbot with Azure OpenAI")
st.markdown("---")
st.markdown("This chatbot answers questions based on documents pre-loaded into `./data_source` and indexed in Chroma DB.")

# --- Initialize Azure OpenAI components and Chroma DB ---
@st.cache_resource # Cache the LLM, embeddings, and vectorstore to avoid re-initialization
def setup_rag_components():
    try:
        # LLM for generation
        llm = AzureChatOpenAI(
            azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_LLM_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            temperature=0.0
        )
        st.sidebar.success("Azure OpenAI Chat LLM connected.")

        # Embedding model for vectorization (MUST match model used for ingestion)
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_EMBEDDING_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            chunk_size=1
        )
        st.sidebar.success("Azure OpenAI Embeddings model connected.")

        # Load Chroma DB
        if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
            st.error(f"Chroma DB not found or empty at `{PERSIST_DIRECTORY}`. Please run `python ingest.py` first to index your documents.")
            st.stop()

        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings, # Important: Use the same embedding function
            collection_name=COLLECTION_NAME
        )
        retriever = vectorstore.as_retriever()
        st.sidebar.success(f"Chroma DB loaded from `{PERSIST_DIRECTORY}`.")
        return llm, retriever
    except Exception as e:
        st.error(f"Error setting up RAG components: {e}")
        st.info("Please verify your Azure OpenAI credentials and deployment names, and ensure `ingest.py` has been run successfully.")
        st.stop()

llm, retriever = setup_rag_components()

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me about the documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        with st.chat_message("assistant"):
            try:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True
                )
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                source_documents = response.get("source_documents", [])

                st.markdown(answer)
                if source_documents:
                    st.markdown("---")
                    st.markdown("### Sources:")
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"- **Source {i+1}**: {doc.page_content[:200]}...") # Show first 200 chars
                        if doc.metadata:
                            st.markdown(f"  **Metadata**: {doc.metadata}")
            except Exception as e:
                st.error(f"An error occurred during response generation: {e}")
                answer = "I apologize, but I couldn't process your request. Please try again."
            st.session_state.messages.append({"role": "assistant", "content": answer})