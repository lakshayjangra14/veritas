import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel

from query_parser import get_query_parser_chain
from reasoning_engine import get_reasoning_chain, FinalAnswer

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

load_dotenv()

# --- Configuration ---
if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "db")

# --- Pydantic Models ---
class Query(BaseModel):
    text: str

# --- Global Variables ---
retriever = None
parser_chain = None
reasoning_chain = None
query_augment_chain = None # NEW

# --- Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever, parser_chain, reasoning_chain, query_augment_chain
    print("--- Loading models and database... ---")
    
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    retriever = db.as_retriever(search_kwargs={"k": 8})
    
    parser_chain = get_query_parser_chain()
    reasoning_chain = get_reasoning_chain()

    chat_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    augment_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Your task is to rephrase the user's query to be more verbose and include likely synonyms found in a medical insurance policy. Only return the rephrased query.\n\nOriginal query: {question}"
    )
    query_augment_chain = augment_prompt | chat_llm | (lambda msg: msg.content if isinstance(msg, AIMessage) else msg)

    print("--- All models and chains loaded successfully. ---")
    yield

app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/query", response_model=FinalAnswer)
async def process_query(query: Query):
    print(f"--- Received Query: {query.text} ---")

    if parser_chain is None or retriever is None or reasoning_chain is None or query_augment_chain is None:
        raise RuntimeError("Parser chain is not initialized. Check if the application startup completed successfully.")
    
    parsed_query = await parser_chain.ainvoke({"query": query.text})
    print(f"--- Step 1: Parsed Query ---\n{parsed_query}")

    # --- NEW: Step 2: Augment the query for better retrieval ---
    augmented_query = await query_augment_chain.ainvoke({"question": query.text})
    print(f"--- Step 2: Augmented Query for Search ---\n{augmented_query}")

    # Step 3: Retrieve docs using the AUGMENTED query
    print("--- Step 3: Retrieving Docs ---")
    relevant_docs = await retriever.ainvoke(str(augmented_query))
    
    print(f"--- DEBUG: RETRIEVED DOCS (Count: {len(relevant_docs)}) ---")

    if not relevant_docs:
        return FinalAnswer(decision="More Info Needed", reasoning="Could not find relevant documents.", justification="No documents found.")

    retrieved_docs_text = "\n\n---\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\n\n{doc.page_content}" for doc in relevant_docs])

    # Step 4: Reason over the retrieved context
    final_answer = await reasoning_chain.ainvoke({
        "parsed_query": str(parsed_query),
        "retrieved_docs_text": retrieved_docs_text
    })
    
    return final_answer