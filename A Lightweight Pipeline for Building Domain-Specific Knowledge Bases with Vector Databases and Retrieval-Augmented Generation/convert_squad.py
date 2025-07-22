import json

with open("train-v1.1.json", "r", encoding="utf-8") as f:
    squad_data = json.load(f)

output_path = "passages.jsonl"
count = 0
limit = 1000  # 先取1000条

with open(output_path, "w", encoding="utf-8") as out_f:
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            text = paragraph["context"].replace("\n", " ").strip()
            if text:
                obj = {"id": str(count), "text": text}
                out_f.write(json.dumps(obj) + "\n")
                count += 1
            if count >= limit:
                break
        if count >= limit:
            break

print(f"Saved {count} passages to {output_path}")
