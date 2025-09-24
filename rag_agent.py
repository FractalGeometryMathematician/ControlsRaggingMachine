# rag_agent.py

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def build_local_t5_pipeline(model_id="google/flan-t5-base", max_new_tokens=768):
    """Builds a local text2text-generation pipeline (Flan-T5)."""
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    gen = pipeline(
        "text2text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
    )
    return HuggingFacePipeline(pipeline=gen)


# Mentor-style verbose answering prompt
MENTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an experienced FRC programming mentor helping rookie teams.\n"
        "Answer in a way that is clear, detailed, and beginner friendly.\n"
        "Use ONLY the context below. If the answer is not in the context, say so.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "Format your response in Markdown with the following sections:\n"
        "## Summary\n"
        "## Step-by-step Instructions\n"
        "## Example Code\n"
        "(Include compilable example code in fenced blocks if possible)\n"
        "## Tips & Common Pitfalls\n"
        "## Cited Sources (filenames or URLs)\n"
    ),
)


def main():
    # --- Step 1: Load embeddings + FAISS index ---
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    print("Loaded FAISS index.")

    # --- Step 2: Retriever (simpler, stricter for relevance) ---
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    # --- Step 3: Local LLM (use flan-t5-base, not small) ---
    llm = build_local_t5_pipeline(model_id="google/flan-t5-base", max_new_tokens=768)

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": MENTOR_PROMPT},
        return_source_documents=True,
    )

    # --- Step 4: Interactive loop ---
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        result = rag_pipeline.invoke({"query": q})

        print("\n================= ANSWER =================")
        print(result["result"])
        print("\n================= SOURCES =================")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "unknown"))
        print("===========================================")


if __name__ == "__main__":
    main()
