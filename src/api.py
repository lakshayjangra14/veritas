import os 
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "db")

class Query(BaseModel):
    text: str

class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict

retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Chroma database not found at {PERSIST_DIRECTORY}")
    print("Loading database and embedding model...")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    print("Database and embedding model loaded successfully.")
    yield
    print("Shutting down the application...")

app = FastAPI(lifespan=lifespan)
# @app.on_event("startup")
# def load_retriever():
#     "This function is called when the FastAPI app starts."
#     global retriever

#     if not os.path.exists(PERSIST_DIRECTORY):
#         raise FileNotFoundError(f"Chroma database not found at {PERSIST_DIRECTORY}")
    
#     print("Loading database and embedding model...")
#     embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

#     db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
#     retriever = db.as_retriever(search_kwargs={"k": 3})

#     print("Database and embedding model loaded successfully.")

# ----- API Endpoints -----

@app.post("/query", response_model=list[DocumentResponse])
async def search_documents(query: Query):
    if not retriever:
        raise RuntimeError("Retriever is not initialized. Please start the server properly.")
    print(f"Received query: {query.text}")
    relevant_docs = retriever.invoke(query.text)

    return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "API is running"}