from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

passages = [json.loads(line) for line in open("passages.jsonl", "r", encoding="utf-8")]
texts = [p["text"] for p in passages]

embeddings = model.encode(texts, batch_size=8, show_progress_bar=True)
np.save("embeddings.npy", embeddings)

print("Done! Saved embeddings.npy.")
