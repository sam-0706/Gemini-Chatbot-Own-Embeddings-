import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core import Settings
from IPython.display import Markdown, display
import os

GOOGLE_API_KEY = "AIzaSyAW1X3ObpQvD_C8097n7x4GfB9UabTfiNo"

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Streamlit title and description
st.title("Gemini-File with Llama-Index")
st.write("This app allows you to upload your own Pdf and query your document, Powered By Gemini")

#function to save a file
def save_uploadedfile(uploadedfile):
     with open(os.path.join("data",uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to directory".format(uploadedfile.name))

# Streamlit input for user file upload
uploaded_pdf = st.file_uploader("Upload your PDF", type=['pdf'])

# Load data and configure the index
if uploaded_pdf is not None:
    input_file = save_uploadedfile(uploaded_pdf)
    st.write("File uploaded successfully!")
    documents = SimpleDirectoryReader("data").load_data()
    Settings.llm = Gemini()
    Settings.embed_model = HuggingFaceEmbedding(model_name = "WhereIsAI/UAE-Large-V1")
    Settings.node_parser = SentenceSplitter(chunk_size=800, chunk_overlap=20)

    # Configure Service Context
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist()
    query_engine = index.as_query_engine()

    # Streamlit input for user query
    user_query = st.text_input("Enter your query:")

    # Query engine with user input
    if user_query:
        response = query_engine.query(user_query)
        st.markdown(f"**Response:** {response}")
else:
    st.write("Please upload a file first.")