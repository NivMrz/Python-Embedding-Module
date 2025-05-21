# Embedding Module

A lightweight Python toolkit for converting documents into searchable vector embeddings. It:

* **Reads** TXT, DOCX & PDF files (auto-detecting file type)
* **Chunks** text by paragraphs, sentences, and fixed-size windows (with overlap)
* **Embeds** each chunk using a SentenceTransformer model (`all-MiniLM-L6-v2`)
* **Indexes** embeddings with FAISS and persists both index and chunk list
* **Searches** the index to retrieve the top-k most similar text chunks for a query

## Installation

Install dependencies from the included `requirements.txt` file:

```bash
pip install -r requirements.txt
```
