# index_builder.py
import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader, TextLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader,
    SitemapLoader, GitLoader, WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEX_DIR = "faiss_index"

def load_local_docs():
    # Load team docs from ./docs (txt/md/pdf etc.)
    docs = []
    if Path("docs").exists():
        docs += DirectoryLoader("docs", glob="**/*.txt", loader_cls=TextLoader).load()
        docs += DirectoryLoader("docs", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader).load()
        # generic loader for other file types (docx, pdf, etc.) if you have unstructured installed
        try:
            docs += DirectoryLoader("docs", glob="**/*.*", loader_cls=UnstructuredFileLoader).load()
        except Exception:
            pass
    return docs

def load_wpilib_docs():
    # Uses sitemap to fetch many pages. Adjust URL if needed.
    # You can add more sitemaps here.
    urls = ["https://docs.wpilib.org/sitemap.xml"]
    docs = []
    for url in urls:
        try:
            loader = SitemapLoader(url, filter_urls=None)
            docs += loader.load()
        except Exception:
            pass
    return docs

def load_ctre_docs():
    # Try main CTRE doc sitemaps; adjust as necessary.
    urls = [
        "https://pro.docs.ctr-electronics.com/sitemap.xml",
        "https://v5.docs.ctr-electronics.com/sitemap.xml",
    ]
    docs = []
    for url in urls:
        try:
            loader = SitemapLoader(url, filter_urls=None)
            docs += loader.load()
        except Exception:
            pass
    return docs

def load_example_code():
    docs = []
    # Clone example repos (public) and index code comments + README
    # Tip: For large repos, you may want to include_only_dirs/include/exclude patterns
    try:
        docs += GitLoader(
            clone_url="https://github.com/wpilibsuite/allwpilib",
            repo_path="tmp_repos/allwpilib",
            branch="main"
        ).load()
    except Exception:
        pass
    # Add your team repo the same way:
    # try:
    #     docs += GitLoader(
    #         clone_url="https://github.com/<your-team>/<your-robot-code>",
    #         repo_path="tmp_repos/team_code",
    #         branch="main"
    #     ).load()
    # except Exception:
    #     pass
    return docs

def chunk_documents(documents):
    # Different chunk sizes for prose vs. code works well; here we keep one simple splitter.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(documents)

def main():
    print("Loading sources...")
    documents = []
    documents += load_local_docs()
    documents += load_wpilib_docs()
    documents += load_ctre_docs()
    documents += load_example_code()

    print(f"Loaded {len(documents)} raw docs")
    if not documents:
        raise SystemExit("No docs found. Add sources or fix loader URLs.")

    print("Chunking...")
    chunks = chunk_documents(documents)
    print(f"Chunks: {len(chunks)}")

    print("Embedding...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    db = FAISS.from_documents(chunks, embeddings)

    print(f"Saving FAISS to {INDEX_DIR}/ ...")
    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    db.save_local(INDEX_DIR)

    print("Done. You can now deploy the index with your app.")

if __name__ == "__main__":
    main()
