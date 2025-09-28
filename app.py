import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
import os

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

# ✅ NEW: Use Mistral through Ollama
llm = OllamaLLM(model="mistral")

def ask(query: str):
    if not query or not query.strip():
        yield "Please enter a question.", ""

    # --- Step 1: Retrieve top documents ---
    retrieved = db.similarity_search_with_score(query, k=3)
    SCORE_THRESHOLD = 0.45

    # --- Step 2: If low retrieval relevance → stream direct LLM ---
    if not retrieved or all(score < SCORE_THRESHOLD for _, score in retrieved):
        print(f"[Fallback] Low retrieval scores → streaming direct LLM for: {query}")
        direct_prompt = f"Question: {query}\nAnswer clearly and concisely."
        partial = ""
        for chunk in llm.stream(direct_prompt):
            partial += chunk
            yield partial, "*(Answered without retrieval)*"
        return

    # --- Step 3: Manual RAG + streaming ---
    # Build context manually from retrieved docs
    context = "\n\n".join(doc.page_content for doc, _ in retrieved)
    rag_prompt = MENTOR_PROMPT.format(context=context, question=query)

    partial = ""
    for chunk in llm.stream(rag_prompt):
        partial += chunk
        sources = "\n".join(f"- {doc.metadata.get('source','unknown')}" for doc, _ in retrieved) or "No sources found."
        yield partial, f"**Sources**:\n{sources}"

with gr.Blocks(title="FRC RAG Mentor") as demo:
    gr.Markdown("# FRC RAG Mentor\nAsk programming questions and get mentor-style answers with sources. Specify language (either java or cpp)")
    q = gr.Textbox(label="Your question", placeholder="e.g., How do I set up a Kraken pivot?")
    go = gr.Button("Ask")
    out_answer = gr.Markdown(label="Answer")
    out_sources = gr.Markdown(label="Sources")

    # ✅ Enable streaming here
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
