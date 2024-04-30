import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def get_file(file_path):
     loader = PdfReader(file_path)
     text = ""
     for page in loader.pages:
         text += page.extract_text()
     return text

def get_text_splitter(document):
     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
     final_document = text_splitter.split_text(document)
     document_gen = [Document(page_content = t) for t in final_document]
     return document_gen

def vectorize_embedding(final_document):
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
    vectors = FAISS.from_documents(final_document, embedding)
    return vectors

def template():
   prompt = ChatPromptTemplate.from_template(
   """
   Hey,you are a brilliant lecturer teaching course in university of bamidele olumilua.
   Your role is to set exam questions for yours students,and please make this examination bsc standard.
   Please use the input to create number of test for the students based on the context.
   Make the test questions based on the content only.

   <context>
   {context}
   <context>
   Input:{input}
   """
   )
   return prompt

def test(model,prompt, vector):
   document_chain = create_stuff_documents_chain(model, prompt)
   retriever = vector.as_retriever()
   retriever_chain = create_retrieval_chain(retriever, document_chain)
   return retriever_chain


llm = ChatGroq(model_name="llama3-70b-8192")

def main():
  st.sidebar.title("File Upload")
  uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
  number = st.number_input("Test number to generate")


  if uploaded_file:
    st.sidebar.success("File uploaded successfully!")
  else:
    st.sidebar.warning("Please upload a PDF file.")

  process_button = st.button("Generate likely question for pdf")

  if process_button and uploaded_file and number:
    document = get_file(uploaded_file)
    text_splitter = get_text_splitter(document)
    embedding = vectorize_embedding(text_splitter)
    prompt_template = template()
    retriever_chain = test(model=llm, prompt = prompt_template ,vector=embedding)
    output = retriever_chain.invoke({"input": str(number)})
    st.write(output["answer"])

  else:
    st.warning("please upload a file and input number of test")

if __name__ == "__main__":
    main()
