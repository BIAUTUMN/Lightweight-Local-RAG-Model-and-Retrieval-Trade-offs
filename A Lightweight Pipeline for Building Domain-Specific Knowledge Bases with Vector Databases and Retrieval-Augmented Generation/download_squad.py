import json
import urllib.request

# SQuAD URL
url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"

print("Downloading SQuAD...")
urllib.request.urlretrieve(url, "train-v1.1.json")
print("Download complete.")
