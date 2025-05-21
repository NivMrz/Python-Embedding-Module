from sentence_transformers import SentenceTransformer
import faiss
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the FAISS vector index database
def load_faiss_index(index_path: str = "faiss.index") -> faiss.Index:
    return faiss.read_index(index_path)

# Load the list of text chunks from a pickle file
def load_chunks(chunks_path: str = "chunks.pkl") -> list[str]:
    with open(chunks_path, "rb") as f:
        return pickle.load(f)

def search_index(
    query: str,
    index: faiss.Index,
    chunks: list[str],
    top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Embed a query, search FAISS index, and return top_k (chunk, distance) pairs.
    """
    q_vec = model.encode([query], show_progress_bar=False).astype('float32')
    distances, indices = index.search(q_vec, top_k)
    return [(chunks[i], float(distances[0][pos])) for pos, i in enumerate(indices[0])]


# Example usage:
if __name__ == "__main__":
    file_path = r"your_file_path"
    my_query = 'your query here'
    chunks = read_and_chunk(file_path)
    embed_and_index(chunks)
    idx = load_faiss_index()
    results = search_index(my_query, idx, ch_list, top_k=2)
    ch_list = load_chunks()
    for chunk, dist in results:
        print(f"{dist:.4f} → {chunk}")
