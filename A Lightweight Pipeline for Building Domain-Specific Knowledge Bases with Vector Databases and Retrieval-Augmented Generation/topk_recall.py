import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 加载数据
df = pd.read_csv("squad_1000.csv")
contexts = df["context"].tolist()
answers = df["answer"].tolist()
questions = df["question"].tolist()

# 加载模型和索引
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("squad_faiss.index")

hit_count = 0
total = len(questions)

for i, q in enumerate(questions):
    q_embed = model.encode([q], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_embed, k=10)

    # 检查答案是否在前三段落里
    retrieved_texts = [contexts[idx].lower() for idx in I[0]]
    answer = str(answers[i]).lower()

    hit = any(answer in text for text in retrieved_texts)
    if hit:
        hit_count += 1

recall = hit_count / total
print(f"\n✅ Top-3 Recall: {recall:.4f} ({hit_count}/{total})")
