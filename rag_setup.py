# rag_setup.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import warnings

# Suppress warnings from LangChain
warnings.filterwarnings("ignore")


# ✅ Load environment variables
load_dotenv()

# ✅ Access keys from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-application"  # Use the exact Pinecone index name you created
MODEL_NAME = "intfloat/multilingual-e5-large"

# ✅ Load & Split Text
#loader_t = TextLoader("data/sample.txt")

# PDF Find and Loader
loader_p = PyPDFLoader("data/doc1.pdf")
documents = loader_p.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = splitter.split_documents(documents)

# ✅ Embedding model
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ Store in Pinecone
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks, 
    embedding=embedding, 
    index_name=INDEX_NAME
)

print("✅ Documents embedded and indexed in Pinecone.")
