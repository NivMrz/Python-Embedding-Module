import os
import sys
import magic
from docx import Document
import PyPDF2
import nltk
import re
from sentence_transformers import SentenceTransformer
import faiss
import pickle


#function to detect file type
def detect_file_type(file_path):
    return magic.from_file(file_path, mime=True)

#function to reaf txt file into text variable
def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

#function to reaf docx file into text variable
def read_docx(file_path):
    doc = Document(file_path)
    return '\n'.join([p.text for p in doc.paragraphs])

#function to reaf pdf file into text variable
def read_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ''
    return text

#function that detects file type and reads it
def read_file(file_path):
    file_type = detect_file_type(file_path)

    if file_type == 'application/pdf':
        return read_pdf(file_path)
    elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return read_docx(file_path)
    elif file_type.startswith('text/'):
        return read_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def chunk_fixed_size(text, chunk_size=200, overlap=100):
    """
        Split text into fixed-size chunks with a specified overlap.
        :param text: Input string
        :param chunk_size: Number of characters per chunk
        :param overlap: Number of overlapping characters between chunks
        :return: List of text chunks
        """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def chunk_by_sentences(text):
        """
        Split text into sentences using a regex.
        Each sentence ends with . ? or ! (or is the trailing piece).
        :param text: Input string
        :return: List of sentences
        """
        pattern = re.compile(r'.*?[\.!?](?=\s|$)', re.DOTALL)
        sentences = [m.group().strip() for m in pattern.finditer(text)]
        # catch any leftover text after the last punctuation
        last = pattern.finditer(text)
        ends = [m.end() for m in last]
        if ends:
            tail = text[ends[-1]:].strip()
            if tail:
                sentences.append(tail)
        return sentences

# 3) Paragraph-boundary chunks
def chunk_by_paragraphs(text):
    """
    Split text into paragraphs (double newlines).
    :param text: Input string
    :return: List of paragraph strings
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def read_and_chunk(
    file_path: str,
    chunk_size: int = 400,
    overlap: int = 150
) -> list[str]:
    """
    Read a file, then produce chunks three ways (paragraphs, sentences, fixed‐size),
    and return a single combined list of all diffrent chunks.
    """
    #Read raw text
    text = read_file(file_path)

    #Chunk by each method
    paras    = chunk_by_paragraphs(text)
    sents    = chunk_by_sentences(text)
    fixed    = chunk_fixed_size(text, chunk_size=chunk_size, overlap=overlap)

    #Combine chunks
    combined = paras + sents + fixed
    return combined

model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_and_index(
    chunks: list[str],
    index_path: str = "faiss.index",
    chunks_path: str = "chunks.pkl"
):
    """
    Embed chunks, build & save a FAISS index, and persist the chunk list.
    """
    # 1) Compute embeddings
    emb_matrix = model.encode(chunks, show_progress_bar=True).astype('float32')

    # 2) Build & save FAISS index
    index = faiss.IndexFlatL2(emb_matrix.shape[1])
    index.add(emb_matrix)
    faiss.write_index(index, index_path)

    # 3) Persist chunk list
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)


if __name__ == '__main__':
    input_path = sys.argv[1]
    chunks = read_and_chunk(input_path)
    embed_and_index(chunks)