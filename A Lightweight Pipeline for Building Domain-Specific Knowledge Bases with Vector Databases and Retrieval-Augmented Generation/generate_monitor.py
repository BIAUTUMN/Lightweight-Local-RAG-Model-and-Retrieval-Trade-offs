from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import json
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import time
import torch
import psutil

# Monitor decorator
def monitor_resources(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        cpu_percent_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        cpu_percent_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)
        print(f"Duration: {end - start:.2f}s")
        print(f"Peak GPU Memory: {peak_memory:.2f} MB")
        print(f"CPU Memory Change: {mem_after - mem_before:.2f} MB")
        print(f"CPU Usage: {cpu_percent_after - cpu_percent_before:.2f}%")
        return result
    return wrapper

# Load passages and embeddings
passages = [json.loads(line) for line in open("passages.jsonl", "r", encoding="utf-8")]
embeddings = np.load("embeddings.npy").astype('float32')
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

queries = json.load(open("queries.json", "r", encoding="utf-8"))

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").half().cuda()

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

@monitor_resources
def generate(inputs):
    return model.generate(**inputs, max_length=64, num_beams=4, early_stopping=True)

all_refs = []
all_preds = []

for q in queries:
    q_vec = embedder.encode(q["question"]).astype('float32')
    D, I = index.search(np.expand_dims(q_vec, axis=0), 5)
    retrieved_text = "\n\n".join([passages[idx]["text"] for idx in I[0]])
    prompt = f"Answer the question based on the context.\n\nContext:\n{retrieved_text}\n\nQuestion:\n{q['question']}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    output = generate(inputs)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    bleu = sentence_bleu([q["answer"].split()], answer.split())
    rouge = scorer.score(q["answer"], answer)['rougeL'].fmeasure
    print("="*50)
    print(f"Q: {q['question']}")
    print(f"A: {answer}")
    print(f"BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}")
    all_refs.append(q["answer"])
    all_preds.append(answer)

P, R, F1 = score(all_preds, all_refs, lang="en")
print("="*50)
print(f"BERTScore F1 mean: {F1.mean().item():.4f}")
