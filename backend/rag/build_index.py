import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    chunks_path = "backend/artifacts/chunks.jsonl"
    index_path = "backend/artifacts/faiss.index"
    meta_path = "backend/artifacts/meta.json"

    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    print("Embedding chunks:", len(texts))

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    vectors = np.array(vectors, dtype="float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, index_path)
    Path(meta_path).write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("✅ Index built successfully")
    print("Vectors:", len(vectors), "Dim:", dim)


if __name__ == "__main__":
    main()
