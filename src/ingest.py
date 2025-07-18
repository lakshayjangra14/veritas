import os 
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables from .env file
load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    print("GOOGLE_API_KEY environment variable is not set.")
    exit()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SOURCE_DOCS_DIR = os.path.join(PROJECT_ROOT, "data", "source_documents")
PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "db")

def load_documents(source_dir: str) -> list:
    """Load documents from the specified directory."""
    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    documents = []
    print(f"Loading {len(pdf_files)} PDF files from {source_dir}...")
    
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(pdf_file)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    
    print(f"Loaded {len(documents)} documents.")
    return documents

def process_and_store_documents():
    """Process documents and store them in a vector database."""
    documents = load_documents(SOURCE_DOCS_DIR)

    if not documents:
        print("No documents found to process.")
        return
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, length_function=len)
    text = text_splitter.split_documents(documents)
    print(f"Split into {len(text)} chunks.")

    print("Creating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    print(f"Creating and persisting vector store in {PERSIST_DIRECTORY}...")

    db = Chroma.from_documents(
        documents=text,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
    )

    print("Ingestion Complete!")
    print(f"Vector store is saved in {PERSIST_DIRECTORY}.")

if __name__ == "__main__":
    script_path = os.path.join("src", "ingest.py")
    process_and_store_documents()