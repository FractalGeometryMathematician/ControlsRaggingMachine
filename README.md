Second version of the RAG tool that is going to be used for purpose of FRC coding. Currently contains a folder of information (3 txt files) with some random lines of information. Code made by chatgpt. Runs from the computer's command prompt. Has an initial prebuilt question, after which users can ask more questions. One python script, rag_agent.py. Another python agent, index_builder.py. This needs to be run once after new documentation is added to the docs folder, as it uses faiss to make more efficient manner of holding data. The new data is stored in the folder faiss_index. Along with this, I had to new files, which are needed in addition to the old ones:

langchain-community

langchain-huggingface

faiss-cpu

sentence-transformers/all-MiniLM-L6-v2 (embeddings)

google/flan-t5-base (LLM, later, more verbose)
