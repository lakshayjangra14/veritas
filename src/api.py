import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Import from our other modules
from src.query_parser import get_query_parser_chain, ParsedQuery

# LangChain and Chroma imports
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv()

# --- Configuration and Path Setup ---
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "db")

# --- Pydantic Data Models ---
class Query(BaseModel):
    text: str

class DocumentResponse(BaseModel):
    page_content: str
    metadata: dict

class ApiResponse(BaseModel):
    parsed_query: ParsedQuery
    retrieved_docs: List[DocumentResponse]

retriever = None
parser_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, parser_chain
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Chroma database not found at {PERSIST_DIRECTORY}")
    
    print("--- Loading models and database... ---")
    
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    parser_chain = get_query_parser_chain()
    
    print("--- Models and database loaded successfully. ---")
    yield
    print("--- Shutting down the application... ---")
    retriever = None
    parser_chain = None

app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/query", response_model=ApiResponse)
async def search_documents(query: Query):
    if not retriever or not parser_chain:
        raise RuntimeError("Models are not initialized.")
    
    print(f"Received query: {query.text}")

    parsed_query = parser_chain.invoke({"query": query.text})
    print(f"Parsed query: {parsed_query}")

    relevant_docs = retriever.invoke(query.text)
    
    response_docs = [DocumentResponse(page_content=doc.page_content, metadata=doc.metadata) for doc in relevant_docs]

    return ApiResponse(parsed_query=parsed_query, retrieved_docs=response_docs)

@app.get("/health")
def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok"}