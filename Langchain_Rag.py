import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Function to read documents
def load_docs(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    return documents

def split_docs(documents, chunk_size=300, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def get_relevant_documents(query, retriever):
    relevant_docs = retriever.get_relevant_documents(query)
    st.title("Relevant Chunks")
    st.write(relevant_docs)
    return relevant_docs

def get_answer(query, chain, retriever):
    relevant_docs = get_relevant_documents(query, retriever)
    response = chain.run(input_documents=relevant_docs, question=query)
    return response

st.title('Question Answering App with Streamlit')

# File upload
uploaded_file = st.file_uploader("Upload a text file", type=['txt'])

if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    
    # Save the uploaded file
    save_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the text from the uploaded file
    documents = load_docs(save_path)

    # Split documents into chunks
    docs = split_docs(documents)

    # Sentence embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a vector store and retriever
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    # Load language model
    llm = Ollama(model="llama3")
    chain = load_qa_chain(llm, chain_type="stuff")

    # Question input
    user_question = st.text_input("Ask a question:")

    if st.button("Submit"):
        answer = get_answer(user_question, chain, retriever)
        st.title("Answer by LLM")
        st.write("Answer:", answer)
