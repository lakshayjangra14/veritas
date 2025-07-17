import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Import from our other modules
from src.query_parser import get_query_parser_chain, ParsedQuery
from src.reasoning_engine import get_reasoning_chain, FinalAnswer

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

retriever = None
parser_chain = None
reasoning_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, parser_chain, reasoning_chain
    if not os.path.exists(PERSIST_DIRECTORY):
        raise FileNotFoundError(f"Chroma database not found at {PERSIST_DIRECTORY}")
    
    print("--- Loading models and database... ---")
    
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    parser_chain = get_query_parser_chain()

    reasoning_chain = get_reasoning_chain()
    
    print("--- Models and database loaded successfully. ---")
    yield
    print("--- Shutting down the application... ---")
    retriever = None
    parser_chain = None
    reasoning_chain = None

app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/query", response_model=FinalAnswer)
async def search_documents(query: Query):
    if retriever is None or parser_chain is None or reasoning_chain is None:
        raise RuntimeError("Models are not initialized.")
    
    print(f"Received query: {query.text}")

    parsed_query = parser_chain.invoke({"query": query.text})
    print(f"Parsed query: {parsed_query}")

    relevant_docs = retriever.invoke(query.text)
    retrieved_docs_txt = "\n\n---\n\n".join(
        [f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\n\n{doc.page_content}" for doc in relevant_docs]
    )
    print(f"Retrieved documents: {retrieved_docs_txt[:500]}...")  
    
    final_answer = reasoning_chain.invoke({
        "parsed_query": str(parsed_query),
        "retrieved_docs_text": retrieved_docs_txt
    })

    print(f"Final answer: {final_answer}")
    return final_answer

@app.get("/health")
def health_check():
    """Provides a simple health check endpoint."""
    return {"status": "ok"}