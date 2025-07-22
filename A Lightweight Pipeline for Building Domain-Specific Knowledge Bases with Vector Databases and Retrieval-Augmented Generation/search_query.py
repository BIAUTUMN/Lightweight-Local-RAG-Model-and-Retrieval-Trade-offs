from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# 1. 加载模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. 加载索引
index = faiss.read_index("squad_faiss.index")

# 3. 加载原始文本
df = pd.read_csv("squad_1000.csv")
contexts = df["context"].tolist()

# 4. 输入查询
query = input("请输入你的查询问题： ")

# 5. 生成查询Embedding
query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

# 6. 搜索
D, I = index.search(query_embedding, k=3)

# 7. 输出结果
print("\n🔍 Top 3 检索结果：\n")
for rank, idx in enumerate(I[0], start=1):
    print(f"Rank {rank}:")
    print(contexts[idx][:300], "...\n")
