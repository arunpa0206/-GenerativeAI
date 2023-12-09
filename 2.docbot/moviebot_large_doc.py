from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# this will load 'YOUR_OPENAI_API_KEY' from the .env file
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')

# mention your path for "Demo-pdf-3-idiots.pdf"
pdf_file_path = 'Demo-pdf-3-idiots.pdf' 

def get_movie_text_from_pdf(pdf_file_path):
  movie_pdf_loader = PyPDFLoader(pdf_file_path)
  # below line uses the load_and_split method of the PyPDFLoader to load the PDF document and split its content into smaller text chunks or documents.
  movie_pdf_documents = movie_pdf_loader.load_and_split() 

  # using CharacterTextSplitter
  movie_pdf_text_splitter = CharacterTextSplitter(separator="\n", # It uses newline characters to separate the text chunks.
                                                  chunk_size=100, # Each text chunk is limited to 100 characters in length.
                                                  chunk_overlap=20) # There is a 20-character overlap between adjacent chunks.

  movie_pdf_text_data = movie_pdf_text_splitter.split_documents(movie_pdf_documents)
  return movie_pdf_text_data

movie_pdf_text_data = get_movie_text_from_pdf(pdf_file_path)


def chunk_and_store_movie_data(movie_pdf_text_data):
  # creates a vector database using Chroma.
  movie_pdf_vectordb = Chroma.from_documents(
    movie_pdf_text_data,
    embedding=OpenAIEmbeddings(), # using embedding provided by OpenAI to represent the text data(movie_pdf_text_data) as vectors.
    persist_directory='./MyDrive' # directory where the vector database(movie_pdf_vectordb) will be persisted or saved.
  )

  movie_pdf_vectordb.persist()
  return movie_pdf_vectordb

movie_pdf_vectordb = chunk_and_store_movie_data(movie_pdf_text_data)

movie_pdf_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# retrieve documents or information from vector database(movie_pdf_vectordb)
retriever = movie_pdf_vectordb.as_retriever() 

movie_pdf_chain = RetrievalQA.from_chain_type(llm = movie_pdf_llm,retriever=retriever)

# write the query describing what you want to ask from the pdf. For example - Here we are asking: 'Brief the story of 3-idiots'
print(movie_pdf_chain({"query": "Brief the story of 3-idiots"}))
