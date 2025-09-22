# rag_agent.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def build_local_t5_pipeline(model_id: str = "google/flan-t5-small", max_new_tokens: int = 256):
    """Build a local seq2seq (text2text) pipeline with Transformers."""
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen = pipeline(
        task="text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=gen)


def make_rag_pipeline():
    # Step 1: Load every .txt in docs/
    loader = DirectoryLoader(
        "docs",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True,
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} docs")
    if not documents:
        raise SystemExit("No documents found in ./docs. Add some .txt files and rerun.")

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3: Embeddings (requires sentence-transformers + torch)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Vector store (FAISS)
    db = FAISS.from_documents(chunks, embeddings)

    # Step 5: Retriever + local LLM
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = build_local_t5_pipeline(model_id="google/flan-t5-small", max_new_tokens=256)

    # RAG chain
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )


def answer_and_show_sources(rag, question: str):
    result = rag.invoke({"query": question})
    print("\nA:", result["result"], "\n")
    print("Top sources:")
    for i, doc in enumerate(result.get("source_documents", []), start=1):
        src = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.strip().replace("\n", " ")
        print(f"[{i}] {src} :: {snippet[:300]}{'...' if len(snippet) > 300 else ''}")


def main():
    rag = make_rag_pipeline()

    # Optional: one starter question so you see output immediately
    starter_q = "What is the difference between retrieval and generation?"
    print(f"\nQ: {starter_q}")
    answer_and_show_sources(rag, starter_q)

    # Interactive loop
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not q:
            continue
        print(f"\nQ: {q}")
        answer_and_show_sources(rag, q)


if __name__ == "__main__":
    main()

