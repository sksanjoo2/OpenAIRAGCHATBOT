import os
from dotenv import load_dotenv

def set_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    print("Environment variables loaded from .env file")

# Load environment variables from .env file
load_dotenv()
DATA_DIRECTORY = "./Data"  # Directory where your local TXT/PDF files are stored
COLLECTION_NAME = "my_rag_collection"
