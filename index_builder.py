# index_builder.py
import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, PyPDFLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"

def load_local_docs():
    """Load only .java, .txt, and .pdf files from ./docs (recursively)."""
    docs = []
    docs_path = Path("docs")

    if not docs_path.exists():
        print("⚠️ No docs/ folder found.")
        return docs

    # --- Load Java files as plain text ---
    java_loader = DirectoryLoader(
        "docs", glob="**/*.java", loader_cls=TextLoader, show_progress=True
    )
    java_docs = java_loader.load()
    docs.extend(java_docs)
    print(f"📄 Loaded {len(java_docs)} Java files")

    # --- Load text files ---
    txt_loader = DirectoryLoader(
        "docs", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
    )
    txt_docs = txt_loader.load()
    docs.extend(txt_docs)
    print(f"📝 Loaded {len(txt_docs)} text files")

    # --- Load PDFs explicitly with PyPDFLoader ---
    pdf_paths = list(docs_path.rglob("*.pdf"))
    for pdf_path in pdf_paths:
        try:
            pdf_docs = PyPDFLoader(str(pdf_path)).load()
            docs.extend(pdf_docs)
            print(f"📚 Loaded PDF: {pdf_path}")
        except Exception as e:
            print(f"⚠️ Failed to load PDF {pdf_path}: {e}")

    return docs


def chunk_documents(documents):
    """Split documents into manageable chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)


def main():
    print("📥 Loading sources...")
    documents = load_local_docs()

    print(f"✅ Loaded {len(documents)} raw docs")
    if not documents:
        raise SystemExit("❌ No docs found. Add sources or fix paths.")

    print("✂️ Chunking...")
    chunks = chunk_documents(documents)
    print(f"📄 Chunks created: {len(chunks)}")

    print("🧠 Embedding...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("📦 Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    print(f"💾 Saving FAISS index to {INDEX_DIR}/ ...")
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    db.save_local(INDEX_DIR)

    print("✅ Done! You can now deploy the index with your app.")


if __name__ == "__main__":
    main()
