from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Load passages and embeddings
passages = [json.loads(line) for line in open("passages.jsonl", "r", encoding="utf-8")]
embeddings = np.load("embeddings.npy").astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 这里放几个测试query
queries = [
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
    {"question": "What is the capital of Australia?", "answer": "Canberra"},
    {"question": "Who discovered penicillin?", "answer": "Alexander Fleming"},
    {"question": "What is the tallest mountain?", "answer": "Mount Everest"},
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ks = [1, 3, 5, 10]

for k in ks:
    hits = 0
    for q in queries:
        q_vec = model.encode(q["question"]).astype('float32')
        D, I = index.search(np.expand_dims(q_vec, axis=0), k)
        retrieved = [passages[idx]["text"] for idx in I[0]]
        if any(q["answer"].lower() in r.lower() for r in retrieved):
            hits += 1
    recall = hits / len(queries)
    print(f"Top-{k} Recall: {recall:.4f}")
