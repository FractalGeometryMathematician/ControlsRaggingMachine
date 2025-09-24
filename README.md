First version of the RAG tool that is going to be used for purpose of FRC coding. Currently contains a folder of information (3 txt files) with some random lines of information. Code made by chatgpt. Runs from the computer's command prompt. Has an initial prebuilt question, after which users can ask more questions. One python script, rag_agent.py. Along with this, I had to download these files using pip: langchain – the main framework you’re using to build your RAG pipeline.

langchain-community – holds many integrations (document loaders, vector stores, etc.).

langchain-huggingface – newer package that contains the Hugging Face LLM/embeddings connectors.

faiss-cpu – vector database library that stores your document embeddings and does similarity search.

transformers – Hugging Face library for running language models locally.

sentence-transformers – library (built on top of transformers) that provides embedding models.

huggingface_hub – client for downloading and caching models from Hugging Face Hub.

torch / torchvision / torchaudio – PyTorch (CPU version) for running neural nets like sentence-transformers and Flan-T5.

and made a hugging face access token and account, the token being stored as an environmental variable in the computer.
