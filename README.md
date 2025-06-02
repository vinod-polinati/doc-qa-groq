# PDF Chatbot 📚

A streamlined PDF question-answering application that allows users to upload PDF documents and ask questions about their content. The application uses TF-IDF vectorization for efficient document similarity search and Groq's LLM API for generating accurate responses.

## Features

- 📄 PDF text extraction and processing
- 🔍 Smart document chunking with overlap
- 📊 TF-IDF-based similarity search
- 💬 Natural language question answering
- 🚀 Efficient memory management
- 🎯 Real-time responses
- 🌐 User-friendly Streamlit interface

## Prerequisites

- Python 3.8 or higher
- A Groq API key ([Get one here](https://console.groq.com))

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Groq API key:
```bash
GROQ_API_KEY=your_api_key_here
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

3. Upload a PDF file using the file uploader

4. Ask questions about the PDF content in the chat interface

## How It Works

1. **PDF Processing**: When a PDF is uploaded, the application extracts text content using PyMuPDF (fitz)

2. **Text Chunking**: The extracted text is split into manageable chunks with overlap for context preservation

3. **Vectorization**: Text chunks are converted to TF-IDF vectors for efficient similarity search

4. **Query Processing**: When a question is asked:
   - The question is vectorized using the same TF-IDF model
   - Most relevant text chunks are retrieved using FAISS similarity search
   - Retrieved context is sent to Groq's LLM for generating a response

## Technical Details

- **Vectorization**: Uses scikit-learn's TfidfVectorizer
- **Similarity Search**: FAISS for efficient vector similarity search
- **LLM Integration**: Groq's llama-3.1-8b-instant model
- **Memory Management**: Implements singleton pattern and proper resource cleanup
- **Concurrency**: Thread-safe implementation with proper locking mechanisms

## Dependencies

- `pymupdf`: PDF processing
- `faiss-cpu`: Vector similarity search
- `scikit-learn`: TF-IDF vectorization
- `groq`: LLM API integration
- `streamlit`: Web interface
- `python-dotenv`: Environment variable management
- `langchain`: LLM chain operations

## Error Handling

The application includes comprehensive error handling for:
- PDF processing issues
- Memory management
- API communication
- Invalid user inputs

## Limitations

- PDF files must be text-based (scanned documents may not work properly)
- Response quality depends on the PDF content quality and chunk size
- Requires active internet connection for LLM API access

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgments

- Groq for providing the LLM API
- Streamlit for the web framework
- FAISS team for the similarity search implementation 