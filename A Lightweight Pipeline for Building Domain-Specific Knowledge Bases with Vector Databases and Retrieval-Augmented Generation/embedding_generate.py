from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# 1. 加载CSV
df = pd.read_csv("squad_1000.csv")

# 2. 初始化模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. 生成Embedding
embeddings = model.encode(
    df["context"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    batch_size=32
)

# 4. 保存Embedding
np.save("squad_embeddings.npy", embeddings)

print("✅ Embedding生成完毕，共生成向量：", embeddings.shape)
