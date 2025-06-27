import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
import shutil
from dotenv import load_dotenv
from config import DATA_DIRECTORY, COLLECTION_NAME, set_environment

# --- Configuration (Azure OpenAI & Data Paths) ---
PERSIST_DIRECTORY = "./chroma_db"  # Directory to store Chroma DB data persistently
load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_EMBEDDING_VERSION", "2024-02-15-preview") # Default if not set
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT") # e.g., "text-embedding-ada-002"

# --- Validate Configuration ---
if not all([AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_EMBEDDING_DEPLOYMENT]):
    print("Error: Missing Azure OpenAI environment variables. Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT.")
    exit()

def ingest_documents():
    print("Starting document ingestion process...")

    # Initialize Embedding Model
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            chunk_size=1 # Recommended for Azure OpenAI embeddings
        )
        print("Azure OpenAI Embeddings initialized successfully.")
    except Exception as e:
        print(f"Error initializing Azure OpenAI Embeddings: {e}")
        print("Please check your AZURE_OPENAI_EMBEDDING_DEPLOYMENT name. It must refer to a deployed *embedding* model (e.g., text-embedding-ada-002).")
        return

    # Load documents from local directory
    print(f"Loading documents from: {DATA_DIRECTORY}")
    try:
        # Use DirectoryLoader for multiple file types
        loader = DirectoryLoader(
            DATA_DIRECTORY,
            glob="**/*.pdf",  # Load all PDF files
            loader_cls=PyPDFLoader,
            silent_errors=True # Continue if some files cause errors
        )
        pdf_documents = loader.load()

        loader = DirectoryLoader(
            DATA_DIRECTORY,
            glob="**/*.txt",  # Load all TXT files
            loader_cls=TextLoader,
            silent_errors=True
        )
        txt_documents = loader.load()

        documents = pdf_documents + txt_documents
        print(f"Loaded {len(documents)} documents (PDFs and TXTs).")

        if not documents:
            print(f"No documents found in {DATA_DIRECTORY}. Please place your files there.")
            return

    except Exception as e:
        print(f"Error loading documents: {e}")
        return

    # Chunking
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} chunks.")

    # Clean existing Chroma DB for a fresh index
    if os.path.exists(PERSIST_DIRECTORY):
        print(f"Clearing existing Chroma DB at: {PERSIST_DIRECTORY}")
        shutil.rmtree(PERSIST_DIRECTORY)
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True) # Ensure directory exists

    # Store in Chroma DB
    print(f"Generating embeddings and storing in Chroma DB at {PERSIST_DIRECTORY}...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=COLLECTION_NAME
    )
    vectorstore.persist() # Ensure data is saved to disk
    print("Chroma DB built and persisted successfully!")

if __name__ == "__main__":
    set_environment()  # Ensure environment variables are set
    print("Environment variables set successfully.")
    print("Starting document ingestion...")
    ingest_documents()