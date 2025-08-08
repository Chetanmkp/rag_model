import os
import warnings
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import requests
import PyPDF2  # Fix undefined name

from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Suppress warnings from LangChain
warnings.filterwarnings("ignore")

# --- 1. Load Environment Variables ---
load_dotenv()

# Set up environment for LangChain
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# --- 2. Application Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "rag-application"
MODEL_NAME = "intfloat/multilingual-e5-large"

# --- 3. Initialize Global Objects (Done once at startup) ---
try:
    # Embedding model
    embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, 
        embedding=embedding
    )
    
    # Setup LLM
    llm = ChatOpenAI(model_name="openai/gpt-4o-mini", temperature=0)
    
    # Create RetrievalQA chain
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")

    print("✅ RAG Chain initialized successfully.")

except Exception as e:
    print(f"❌ Error during initialization: {e}")
    rag_chain = None # Set to None if initialization fails

# --- 4. FastAPI Application ---
app = FastAPI(title="Retrieval-Augmented Generation API")

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

@app.get("/", summary="Check if API is running")
def read_root():
    return {"status": "ok", "message": "RAG API is running."}

@app.post("/query", response_model=QueryResponse, summary="Ask a question to the RAG model")
def ask_question(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is not initialized. Check server logs for errors.")
    
    try:
        result = rag_chain.run(request.question)
        return QueryResponse(answer=result)
    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

@app.get("/query/{question}", response_model=QueryResponse, summary="Ask a question to the RAG model via URL")
def ask_question_via_url(question: str):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain is not initialized. Check server logs for errors.")

    try:
        result = rag_chain.run(question)
        return QueryResponse(answer=result)
    except Exception as e:
        print(f"❌ Error during query processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to process the query.")

@app.post("/hackrx/run", response_model=HackRxResponse, summary="Run Submissions")
def run_submissions(request: HackRxRequest, authorization: Optional[str] = Header(None)):
    if authorization != "Bearer e33ecffb686ac4409ef84a4cebb13b83e58b120976b3e7b73481e7cc3daf20de":
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Download the document
    try:
        response = requests.get(request.documents)
        response.raise_for_status()
        with open("temp_policy.pdf", "wb") as f:
            f.write(response.content)
    except Exception as e:
        print(f"❌ Error downloading document: {e}")
        raise HTTPException(status_code=400, detail="Failed to download the document.")

    # Extract text from the downloaded PDF
    try:
        with open("temp_policy.pdf", "rb") as file:
            reader = PyPDF2.PdfReader(file)
            content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"❌ Error extracting text from PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract text from the document.")

    # Split text into chunks
    chunks = split_by_tokens(content, max_tokens=3000)

    # Process each question
    answers = []
    for question in request.questions:
        best_chunk = find_most_relevant_chunk(chunks, question)
        prompt = f"""
You are a Retrieval-Augmented Generation (RAG) assistant. Follow these rules EXACTLY:

1. READ the context below carefully
2. ONLY use information that is EXPLICITLY stated in the context
3. If the question asks for information NOT in the context, respond: "I cannot answer this question as the information is not available in my knowledge base."
4. Do NOT add any external knowledge or common facts
5. Quote directly from the context when possible

CONTEXT (this is your ONLY source of information):
--------------------
{best_chunk}
--------------------

Based ONLY on the context above, answer this question:
QUESTION: {question}

If the answer is not explicitly in the context, say you cannot answer it.

ANSWER:
"""
        try:
            headers = {
                "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "openai/gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 100  # Reduced from 200 to 100
            }
            response = requests.post(f"{os.getenv('OPENAI_API_BASE')}/chat/completions", json=payload, headers=headers)
            if response.status_code == 200:
                answer = response.json()["choices"][0]["message"]["content"].strip()
                answers.append(answer)
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                answers.append("Error processing the question.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            answers.append("Error processing the question.")

    return HackRxResponse(answers=answers)

def split_by_tokens(text, max_tokens=3000, model="gpt-4o-mini"):
    """Split text into chunks by token count."""
    import tiktoken
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    while tokens:
        chunk = tokens[:max_tokens]
        chunks.append(enc.decode(chunk))
        tokens = tokens[max_tokens:]
    return chunks

def find_most_relevant_chunk(chunks, question):
    """Find the chunk with the most question word matches."""
    question_words = set(question.lower().split())
    best_chunk = ""
    max_matches = 0
    for chunk in chunks:
        matches = sum(1 for word in question_words if word in chunk.lower())
        if matches > max_matches:
            max_matches = matches
            best_chunk = chunk
    return best_chunk
