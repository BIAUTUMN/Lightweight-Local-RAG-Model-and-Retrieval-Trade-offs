import faiss
import numpy as np

# 1. 加载Embedding
embeddings = np.load("squad_embeddings.npy").astype("float32")

# 2. 创建索引
index = faiss.IndexFlatL2(embeddings.shape[1])

# 3. 添加向量
index.add(embeddings)

# 4. 保存索引
faiss.write_index(index, "squad_faiss.index")

print("✅ FAISS索引创建完毕！共收录向量：", index.ntotal)
