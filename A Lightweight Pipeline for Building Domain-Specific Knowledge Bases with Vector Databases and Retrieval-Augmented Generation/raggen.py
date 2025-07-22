from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
import numpy as np
import pandas as pd

# 1. 加载模型
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")

# 2. 加载索引
index = faiss.read_index("squad_faiss.index")

# 3. 加载上下文
df = pd.read_csv("squad_1000.csv")
contexts = df["context"].tolist()

while True:
    # 4. 输入问题
    query = input("\n请输入你的查询问题 (输入 exit 退出): ")
    if query.strip().lower() == "exit":
        break

    # 5. 检索
    query_embedding = retriever_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(query_embedding, k=10)

    retrieved = [contexts[idx] for idx in I[0]]

    # 6. 拼接Prompt
    context_text = " ".join(retrieved)
    prompt = f"Answer the question based on the context.\n\nContext: {context_text}\n\nQuestion: {query}\n\nAnswer:"

    # 7. Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # 8. Generate
    output = t5_model.generate(**inputs, max_length=64)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print("\n💡 RAG生成的回答：", answer)
