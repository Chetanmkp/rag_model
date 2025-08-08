import os
import requests
import PyPDF2
from dotenv import load_dotenv
import tiktoken

# === Load environment variables ===
load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")

# === List your PDF files here ===
pdf_paths = [
    # "data/file1.pdf",
    # "data/file2.pdf",
    # "data/file3.pdf",
    # "data/file4.pdf",
    # "data/file5.pdf"
    "data/doc1.pdf",
]

# === Extract text from all PDFs ===
def extract_text_from_pdfs(file_paths):
    combined_text = ""
    for path in file_paths:
        if not os.path.exists(path):
            print(f"❌ Error: File '{path}' not found.")
            continue
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    combined_text += content + "\n"
    return combined_text.strip()

# === Split text into chunks by token count ===
def split_by_tokens(text, max_tokens=3000, model="gpt-4o-mini"):
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []
    while tokens:
        chunk = tokens[:max_tokens]
        chunks.append(enc.decode(chunk))
        tokens = tokens[max_tokens:]
    return chunks

# === Simple relevance filter: find chunk with most question words ===
def find_most_relevant_chunk(chunks, question):
    question_words = set(question.lower().split())
    best_chunk = ""
    max_matches = 0
    for chunk in chunks:
        matches = sum(1 for word in question_words if word in chunk.lower())
        if matches > max_matches:
            max_matches = matches
            best_chunk = chunk
    return best_chunk

# === Validate API key ===
def validate_api_key():
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in .env file")
        return False
    return True

# === MAIN ===
if not validate_api_key():
    exit(1)

context_data = extract_text_from_pdfs(pdf_paths)
if not context_data:
    print("❌ No content extracted from PDFs.")
    exit(1)

print("📄 All PDFs loaded.")
chunks = split_by_tokens(context_data, max_tokens=3000)
print(f"✅ Context split into {len(chunks)} chunks.")
print("🤖 RAG System Ready! Type 'exit' to quit.")

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    best_chunk = find_most_relevant_chunk(chunks, query)

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
QUESTION: {query}

If the answer is not explicitly in the context, say you cannot answer it.

ANSWER:
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 200
    }

    try:
        response = requests.post(f"{api_base}/chat/completions", json=payload, headers=headers)
        if response.status_code == 200:
            answer = response.json()["choices"][0]["message"]["content"]
            print(f"\n🤖 Answer:\n{answer.strip()}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
