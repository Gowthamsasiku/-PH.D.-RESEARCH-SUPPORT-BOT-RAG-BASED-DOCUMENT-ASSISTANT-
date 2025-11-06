# Research Support Bot with RAG

A powerful document assistant that helps researchers find answers in their PDF documents using Retrieval-Augmented Generation (RAG) with Google's Gemini model.

![Research Support Bot](https://img.shields.io/badge/Status-Active-success)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Features

- **Document Processing**: Automatically processes PDF documents from a specified folder
- **Semantic Search**: Finds relevant information using vector similarity search
- **AI-Powered Answers**: Generates accurate answers using Google's Gemini model
- **Source Citation**: Shows the source document and exact location of information
- **User-Friendly Interface**: Simple and intuitive web interface built with Streamlit

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Google API key with access to Gemini models
- PDF documents to analyze (place them in the `pdfs` folder)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-support-bot.git
   cd research-support-bot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv myenv
   .\myenv\Scripts\activate  # On Windows
   # OR
   source myenv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the project root with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

### Usage

1. **Add your PDFs**
   - Place your PDF documents in the `pdfs` folder
   - The system will process all PDFs in this directory

2. **Run the application**
   ```bash
   streamlit run app.py
   ```

3. **Access the web interface**
   - Open your browser and go to `http://localhost:8501`
   - Click "Initialize System" in the sidebar to process your documents
   - Start asking questions about your research documents

## 🛠️ Project Structure

```
research-support-bot/
├── app.py                 # Main Streamlit application
├── main.py                # Core RAG implementation
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this file)
├── pdfs/                 # Directory for PDF documents (create this folder)
└── chroma_db/            # Vector database storage (auto-created)
```

## 🤖 How It Works

1. **Document Loading**: The system loads PDFs from the `pdfs` directory
2. **Text Processing**: Documents are split into manageable chunks
3. **Vector Embedding**: Text chunks are converted to vector embeddings using HuggingFace's sentence-transformers
4. **Vector Storage**: Embeddings are stored in ChromaDB for efficient similarity search
5. **Question Answering**: When you ask a question, the system:
   - Converts the question to an embedding
   - Finds the most relevant document chunks
   - Uses Google's Gemini to generate an answer based on the context

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your Google API key is correctly set in the `.env` file
   - Verify the key has access to the Gemini API

2. **PDF Loading Issues**
   - Make sure PDFs are not password protected
   - Check that the `pdfs` directory exists and contains PDF files

3. **Dependency Conflicts**
   - Use the exact versions specified in `requirements.txt`
   - Create a fresh virtual environment if you encounter conflicts

### Getting Help

If you encounter any issues, please:
1. Check the terminal for error messages
2. Ensure all dependencies are installed correctly
3. Verify your PDF files are accessible and not corrupted

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the web interface
- [LangChain](https://www.langchain.com/) for RAG implementation
- [Chroma](https://www.trychroma.com/) for vector storage
- [Google Gemini](https://ai.google.dev/) for generative AI capabilities
- [HuggingFace](https://huggingface.co/) for sentence transformers

---

<div align="center">
  Made with ❤️ for researchers everywhere
</div>
