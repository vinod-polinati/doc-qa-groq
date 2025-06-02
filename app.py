import streamlit as st
from pdf_processor import PDFProcessor
import os
from dotenv import load_dotenv

load_dotenv()

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    st.error("Please set your GROQ_API_KEY in the .env file")
    st.stop()

# Initialize PDFProcessor (singleton)
if "pdf_processor" not in st.session_state:
    try:
        st.session_state.pdf_processor = PDFProcessor()
    except Exception as e:
        st.error(f"Error initializing PDF processor: {str(e)}")
        st.stop()

st.title("📚 PDF Chatbot")
st.write("Upload a PDF and ask questions about its content!")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Process PDF when uploaded
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        try:
            with st.spinner("Processing PDF..."):
                st.session_state.pdf_processor.process_pdf_and_create_index(uploaded_file)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chat_history = []  # Clear chat history for new file
            st.success("PDF processed successfully!")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.stop()

    # Chat interface
    st.subheader("Chat")
    
    # Display chat history
    for q, a in st.session_state.get('chat_history', []):
        st.write("🙋‍♂️ **You:** " + q)
        st.write("🤖 **Assistant:** " + a)
        st.write("---")

    # Query input
    query = st.text_input("Ask a question about the PDF:")
    if st.button("Ask"):
        if query:
            try:
                with st.spinner("Thinking..."):
                    answer = st.session_state.pdf_processor.query_pdf(query)
                    if not hasattr(st.session_state, 'chat_history'):
                        st.session_state.chat_history = []
                    st.session_state.chat_history.append((query, answer))
                    
                    # Display the new Q&A
                    st.write("🙋‍♂️ **You:** " + query)
                    st.write("🤖 **Assistant:** " + answer)
                    st.write("---")
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
else:
    st.info("Please upload a PDF file to begin.") 