import json

with open("train-v1.1.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

queries = []
count = 0
limit = 100  # 先抽100条

for article in squad_data["data"]:
    for paragraph in article["paragraphs"]:
        qas = paragraph["qas"]
        for qa in qas:
            question = qa["question"]
            answer = qa["answers"][0]["text"] if qa["answers"] else ""
            if question and answer:
                queries.append({"question": question.strip(), "answer": answer.strip()})
                count += 1
            if count >= limit:
                break
        if count >= limit:
            break

with open("queries.json", "w", encoding="utf-8") as f_out:
    json.dump(queries, f_out, indent=2, ensure_ascii=False)

print(f"Saved {count} queries to queries.json.")
