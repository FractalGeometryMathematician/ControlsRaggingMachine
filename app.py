import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM   # 👈 NEW
import os

# --- Remove the old T5 pipeline completely ---

MENTOR_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an experienced FRC programming mentor helping rookie teams.\n"
        "Answer in a way that is clear, detailed, and beginner friendly.\n"
        "Use ONLY the context below. If the answer is not in the context, say so.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "Respond in Markdown with these sections:\n"
        "## Summary\n"
        "## Step-by-step Instructions\n"
        "## Example Code\n"
        "## Tips & Common Pitfalls\n"
        "## Cited Sources\n"
    ),
)

# ---- Load FAISS index ----
INDEX_DIR = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.isdir(INDEX_DIR):
    raise SystemExit(
        "Missing faiss_index/. Build locally with index_builder.py and commit the folder "
        "(index.faiss + index.pkl)."
    )

db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ✅ NEW: Use Mistral through Ollama
llm = OllamaLLM(model="mistral")

rag = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": MENTOR_PROMPT},
    return_source_documents=True,
)

def ask(query: str):
    if not query or not query.strip():
        return "Please enter a question.", ""

    res = rag.invoke({"query": query})
    answer = res.get("result", "").strip()
    source_docs = res.get("source_documents", [])

    if not answer or answer.lower().startswith("i don't know"):
        return (
            "❓ Sorry, I couldn’t find anything about that in the training docs.",
            ""
        )

    sources = "\n".join(
        f"- {doc.metadata.get('source','unknown')}" for doc in source_docs
    ) or "No sources found."

    return answer, f"**Sources**:\n{sources}"


with gr.Blocks(title="FRC RAG Mentor") as demo:
    gr.Markdown("# FRC RAG Mentor\nAsk programming questions and get mentor-style answers with sources.")
    q = gr.Textbox(label="Your question", placeholder="e.g., How do I set up a Kraken pivot?")
    go = gr.Button("Ask")
    out_answer = gr.Markdown(label="Answer")
    out_sources = gr.Markdown(label="Sources")

    go.click(fn=ask, inputs=q, outputs=[out_answer, out_sources], show_progress="full")
    q.submit(fn=ask, inputs=q, outputs=[out_answer, out_sources], show_progress="full")

    demo.load(
        fn=lambda: ("👋 Model warmed up, ready for your first question!", ""),
        inputs=None,
        outputs=[out_answer, out_sources],
        show_progress="full"
    )

if __name__ == "__main__":
    demo.launch()
