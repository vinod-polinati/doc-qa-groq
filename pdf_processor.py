import os
from typing import List, Dict
import fitz  # PyMuPDF
import numpy as np
from dotenv import load_dotenv
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import groq
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc
import threading
import multiprocessing

# Set multiprocessing start method
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

load_dotenv()

class PDFProcessor:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PDFProcessor, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            try:
                # Initialize vectorizer
                self.vectorizer = TfidfVectorizer(
                    lowercase=True,
                    strip_accents='unicode',
                    ngram_range=(1, 2)
                )
                
                # Initialize other components
                self.chunks = []
                self.index = None
                self.embeddings = None
                
                # Initialize Groq client
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                
                os.environ["GROQ_API_KEY"] = api_key
                self.groq_client = groq.Client()
                
                self._initialized = True
            except Exception as e:
                raise RuntimeError(f"Error initializing PDFProcessor: {str(e)}")
    
    def extract_text(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            text = " ".join(page.get_text() for page in doc)
            doc.close()
            return text
        except Exception as e:
            raise RuntimeError(f"Error extracting text from PDF: {str(e)}")
    
    def create_chunks(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks with smaller size"""
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
            self.chunks = text_splitter.split_text(text)
            return self.chunks
        except Exception as e:
            raise RuntimeError(f"Error creating text chunks: {str(e)}")
    
    def create_embeddings_index(self):
        """Create FAISS index from text chunks using TF-IDF"""
        try:
            # Clean up any existing index
            if self.index is not None:
                del self.index
                gc.collect()
            
            # Create TF-IDF vectors
            self.embeddings = self.vectorizer.fit_transform(self.chunks)
            embeddings_dense = self.embeddings.toarray().astype('float32')
            
            # Create new index
            dimension = embeddings_dense.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings_dense)
            
            # Clear memory
            del embeddings_dense
            gc.collect()
            
        except Exception as e:
            raise RuntimeError(f"Error creating embeddings index: {str(e)}")
    
    def get_relevant_chunks(self, query: str, k: int = 3) -> List[str]:
        """Get most relevant chunks for a query"""
        try:
            # Transform query using the same vectorizer
            query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
            distances, indices = self.index.search(query_vector, k)
            return [self.chunks[i] for i in indices[0]]
        except Exception as e:
            raise RuntimeError(f"Error retrieving relevant chunks: {str(e)}")
    
    def get_answer(self, query: str, context: List[str]) -> str:
        """Get answer using Groq API"""
        prompt = f"""You are a helpful assistant that answers questions based on the provided context.
        Always base your answers on the context provided. If you cannot find the answer in the context, say so.
        
        Context: {' '.join(context)}
        
        Question: {query}
        
        Answer:"""
        
        try:
            completion = self.groq_client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=1024
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error getting response from Groq: {str(e)}"
    
    def process_pdf_and_create_index(self, pdf_file):
        """Process PDF file and create search index"""
        try:
            # Clear memory before processing
            gc.collect()
            
            text = self.extract_text(pdf_file)
            self.create_chunks(text)
            self.create_embeddings_index()
        except Exception as e:
            raise RuntimeError(f"Error processing PDF: {str(e)}")
    
    def query_pdf(self, query: str) -> str:
        """Query the PDF content"""
        try:
            if not self.index:
                return "Please upload a PDF file first."
            
            relevant_chunks = self.get_relevant_chunks(query)
            answer = self.get_answer(query, relevant_chunks)
            return answer
        except Exception as e:
            return f"Error processing query: {str(e)}"
    
    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if hasattr(self, 'index') and self.index is not None:
                del self.index
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                del self.embeddings
            gc.collect()
        except:
            pass 