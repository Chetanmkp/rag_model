# rag_query.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import warnings

# Suppress warnings from LangChain
warnings.filterwarnings("ignore")


# ✅ Load .env variables
load_dotenv()

# ✅ OpenRouter setup (OpenAI-compatible)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"  # OpenRouter endpoint

# Configs
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-application"
MODEL_NAME = "intfloat/multilingual-e5-large"

# ✅ Embedding model
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# ✅ Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embedding
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 24})

# ✅ Setup OpenRouter-compatible LLM
llm = ChatOpenAI(model_name="openai/gpt-4o-mini", temperature=0)

# ✅ Create RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

# ✅ Ask your query
query = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
]

for i in query:
    result = rag_chain.run(i)
    print("\n📌 Answer:\n", result)
