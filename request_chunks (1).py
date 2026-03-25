# import requests
# import json
# import os
# import pandas as pd
# def create_embedding(text_list):
#     r=requests.post("http://localhost:11434/api/embed",json={
#         "model":"bge-m3",
#         "input":text_list
#     })
#     embedding=r.json()['embeddings']
#     return embedding
# jsons=os.listdir("jsons")
# my_dicts=[]
# chunk_id=0
# for json_file in jsons:
#     with open(f"jsons/{json_file}") as f:
#         content=json.load(f)
#     print(f"Creating embeddings for {json_file}")
#     embeddings=create_embedding([c['text'] for c in content['chunks']])
#     print("Before senond for loop")
#     for i, chunk in enumerate(content['chunks']):
#         chunk['chunk_id'] = chunk_id
#         chunk['embedding'] = embeddings[i]
#         chunk_id += 1
#         my_dicts.append(chunk)
# df = pd.DataFrame.from_records(my_dicts)
# print(df)
import requests
import os
import json
import pandas as pd
import numpy as np
import joblib
def create_embedding(text_list):
    r = requests.post( "http://127.0.0.1:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
    data = r.json()
    if "embeddings" not in data:
        raise RuntimeError(f"Invalid response: {data}")
    return data["embeddings"]
def batch(iterable, size=5):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]
jsons = os.listdir("jsons")  # List all the jsons 
my_dicts = []
chunk_id = 0
for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    texts = [c["text"].strip()
        for c in content["chunks"]
        if isinstance(c.get("text"), str) and c["text"].strip()
    ]
    all_embeddings = []

    for text_batch in batch(texts, size=5):
        try:
            all_embeddings.extend(create_embedding(text_batch))
        except RuntimeError as e:
            print(" Skipping failed batch:", e)
    valid_chunks = [
        c for c in content["chunks"]
        if isinstance(c.get("text"), str) and c["text"].strip()
    ]
    for chunk, embedding in zip(valid_chunks, all_embeddings):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedding
        chunk_id += 1
        my_dicts.append(chunk)
df = pd.DataFrame.from_records(my_dicts)
joblib.dump(df, 'embeddings.joblib')