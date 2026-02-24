import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings # Simpler for local use
import config

def main():
    print("Checking for documents...")
    all_pages = []
    doc_folder = "./documents"

    if not os.path.exists(doc_folder):
        os.makedirs(doc_folder)
        print("Created folder")
        return

    for filename in os.listdir(doc_folder):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(doc_folder, filename))
            all_pages.extend(loader.load())

    print(f"Splitting {len(all_pages)} pages into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(all_pages)

    embeddings = OllamaEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.EMBEDDING_BASE_URL
    )

    print("Saving to ChromaDB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db",
        collection_name=config.COLLECTION_NAME
    )
    print(f"saved {len(chunks)} chunks.")

if __name__ == "__main__":
    main()
