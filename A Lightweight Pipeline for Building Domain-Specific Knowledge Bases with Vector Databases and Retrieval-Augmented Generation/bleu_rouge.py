from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import faiss
import numpy as np
import pandas as pd

# 加载数据
df = pd.read_csv("squad_1000.csv")
contexts = df["context"].tolist()
answers = df["answer"].tolist()
questions = df["question"].tolist()

# 模型
retriever = SentenceTransformer("all-MiniLM-L6-v2")
t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
index = faiss.read_index("squad_faiss.index")
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

total_bleu = 0
total_rouge = 0
count = 0

# 限制前N条
N = 100

for i in range(N):
    q = questions[i]
    ref = answers[i]

    q_embed = retriever.encode([q], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_embed, k=10)
    retrieved = [contexts[idx] for idx in I[0]]
    context_text = " ".join(retrieved)

    prompt = f"Answer the question based on the context.\n\nContext: {context_text}\n\nQuestion: {q}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = t5.generate(**inputs, max_length=64)
    pred = tokenizer.decode(output[0], skip_special_tokens=True)

    bleu = sentence_bleu([ref.split()], pred.split())
    rouge = scorer.score(ref, pred)['rougeL'].fmeasure

    total_bleu += bleu
    total_rouge += rouge
    count += 1

avg_bleu = total_bleu / count
avg_rouge = total_rouge / count

print(f"\n✅ Evaluated {count} samples")
print(f"Average BLEU: {avg_bleu:.4f}")
print(f"Average ROUGE-L F1: {avg_rouge:.4f}")
