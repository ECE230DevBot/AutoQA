from corelibs.embedding_model import Embedding_Model
import time

print("BEGIN")
with open('api_key.txt') as f:
    api_key = f.read().strip()
#model = Embedding_Model(url = "https://api.mistral.ai/v1/embeddings", api_key=api_key, model="mistral-embed")
model = Embedding_Model(url = 'https://api.openai.com/v1/embeddings', api_key=api_key, model="text-embedding-3-large")
chunks = ["hello world", 'whats up', 'go kys', "um, what?", "the library is", "my cat is snoring", "the world war", "america has always been"]
embeddings = model.setup_doc_embeds(chunks, override_saves=True)
time.sleep(3)
search = model.search("pets", 2)
retrieved_chunk = [chunks[i] for i in search[1].tolist()[0]]
print(retrieved_chunk)
print("FINISH")