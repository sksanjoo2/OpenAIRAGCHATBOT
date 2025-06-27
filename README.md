# RAG LLM Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangChain, Azure OpenAI, and Streamlit. This chatbot can ingest documents (PDF and TXT files) and answer questions based on the content using Azure OpenAI's GPT and embedding models.

## Features

- 📄 **Document Ingestion**: Support for PDF and TXT files
- 🔍 **Vector Search**: Uses Chroma DB for efficient document retrieval
- 🤖 **Azure OpenAI Integration**: Leverages GPT models for chat and embedding models for document vectorization
- 🎯 **RAG Pipeline**: Combines retrieval and generation for accurate, context-aware responses
- 🖥️ **Streamlit UI**: User-friendly web interface
- 🔧 **Configurable**: Easy setup with environment variables

## Project Structure

```
RAGLLMCHATBOT/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
├── config.py            # Configuration settings
├── ingest.py            # Document ingestion script
├── chatbot.py           # Main Streamlit chatbot application
├── Data/                # Your documents go here (create this folder)
│   ├── document1.pdf
│   ├── document2.txt
│   └── ...
└── chroma_db/           # Vector database (auto-created)
```

## Setup Instructions

### 1. Prerequisites

- Python 3.8 or higher
- Azure OpenAI service with deployed models
- Git (optional)

### 2. Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd RAGLLMCHATBOT
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 3. Configuration

#### Create Data Directory
Create a `Data` folder in the project root and add your documents:
```bash
mkdir Data
# Copy your PDF and TXT files to the Data folder
```

#### Set up Environment Variables
Create a `.env` file in the project root with your Azure OpenAI credentials:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT="https://your-resource-name.openai.azure.com"
AZURE_OPENAI_API_KEY="your-api-key-here"

# Model Deployments (use your actual deployment names)
AZURE_OPENAI_LLM_DEPLOYMENT="gpt-35-turbo"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-ada-002"

# API Versions
AZURE_OPENAI_API_LLM_VERSION="2024-02-15-preview"
AZURE_OPENAI_API_EMBEDDING_VERSION="2023-05-15"
```

**How to get Azure OpenAI credentials:**
1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Go to "Keys and Endpoint" section
4. Copy the endpoint URL and one of the keys
5. Note your model deployment names from the "Model deployments" section

### 4. Usage

#### Step 1: Ingest Documents
First, add your documents to the `Data` folder, then run the ingestion script:

```bash
python ingest.py
```

This will:
- Load all PDF and TXT files from the `Data` folder
- Split documents into chunks
- Generate embeddings using Azure OpenAI
- Store embeddings in Chroma vector database

#### Step 2: Run the Chatbot
Start the Streamlit application:

```bash
streamlit run chatbot.py
```

The chatbot will be available at `http://localhost:8501`

## Configuration Details

### Supported File Types
- **PDF files**: `.pdf`
- **Text files**: `.txt`

### Customizable Settings
You can modify these settings in `config.py`:
- `DATA_DIRECTORY`: Path to your documents folder
- `COLLECTION_NAME`: Name for the Chroma collection

### Model Configuration
The system uses:
- **Chat Model**: Azure OpenAI GPT models for generating responses
- **Embedding Model**: Azure OpenAI embedding models for document vectorization
- **Vector Store**: Chroma DB for storing and retrieving document embeddings

## Troubleshooting

### Common Issues

1. **Import Error for config variables**
   - Ensure `config.py` contains `DATA_DIRECTORY`, `COLLECTION_NAME`, and `set_environment` function

2. **Missing Azure OpenAI environment variables**
   - Check that all required variables are set in `.env`
   - Verify your Azure OpenAI endpoint and API key
   - Ensure model deployment names match your Azure setup

3. **No documents found**
   - Verify documents are in the `Data` folder
   - Check file formats (only PDF and TXT are supported)
   - Ensure proper file permissions

4. **Chroma DB issues**
   - Delete the `chroma_db` folder and re-run `python ingest.py`
   - Ensure sufficient disk space

### Debug Mode
To enable verbose logging, you can modify the scripts to include debug information.

## Dependencies

Key packages used:
- `streamlit`: Web interface
- `langchain`: RAG framework
- `langchain-openai`: Azure OpenAI integration
- `langchain-chroma`: Vector database
- `langchain-community`: Document loaders
- `python-dotenv`: Environment variable management

## Security Notes

- Never commit your `.env` file to version control
- Keep your Azure OpenAI API keys secure
- The `.gitignore` file is configured to exclude sensitive files

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Azure OpenAI documentation
3. Check LangChain documentation
4. Create an issue in the repository

---

**Note**: This chatbot requires an active Azure OpenAI subscription and deployed models. Make sure you have the necessary permissions and quotas set up in your Azure account.