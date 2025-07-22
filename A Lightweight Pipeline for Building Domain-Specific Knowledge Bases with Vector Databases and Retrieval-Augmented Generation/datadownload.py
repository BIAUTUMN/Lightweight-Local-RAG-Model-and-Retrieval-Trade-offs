from datasets import load_dataset
import pandas as pd

# 加载SQuAD前1000条
dataset = load_dataset("squad", split="train[:1000]")

# 提取字段
contexts = []
questions = []
answers = []

for item in dataset:
    contexts.append(item["context"])
    questions.append(item["question"])
    # 如果没有答案，给空字符串
    if len(item["answers"]["text"]) > 0:
        answers.append(item["answers"]["text"][0])
    else:
        answers.append("")

# 构造成DataFrame
df = pd.DataFrame({
    "context": contexts,
    "question": questions,
    "answer": answers
})

# 保存为CSV
df.to_csv("squad_1000.csv", index=False, encoding="utf-8")

print("✅ SQuAD前1000条数据已保存为 squad_1000.csv")
