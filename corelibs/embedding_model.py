from tqdm import tqdm
import requests
import pickle
import os
import json
import faiss
import numpy as np
import time

class Embedding_Model:
    url: str # URL of the embeddings endpoint
    api_key: str # API key for the endpoint
    model: str # name of model to use for embeddings
    embeddings_save_path : str
    index : faiss.IndexFlatL2
    
    def __init__(self, url : str = None, model : str = None, api_key : str = None, embeddings_save_path : str = "corelibs/embeddings/embedding.pkl"):
        self.url = url
        self.model = model
        self.api_key = api_key
        self.embeddings_save_path = embeddings_save_path
        
    def get_embed(self, inputs : list[str]):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        data = {
            "input": inputs,
            "model": self.model,
            "encoding_format": "float"
        }
        response = requests.post(self.url, headers=headers, json=data)
        response = json.loads(response.text)
        response = response['data']
        response = np.array(list(map(lambda x: x['embedding'], response)))
        return response
        
    def setup_doc_embeds(self, inputs : list[str], override_saves : bool = True):
        if not os.path.exists(self.embeddings_save_path) or override_saves:
            # Batch process because this thing sucks
            doc_embeddings = None
            for i in tqdm(range(0, len(inputs), 10)):
                ran = False
                while ran == False:
                    try:
                        if doc_embeddings is None:
                            doc_embeddings = self.get_embed(inputs[i:i+10])
                        else:
                            doc_embeddings = np.concatenate((doc_embeddings, self.get_embed(inputs[i:i+10])))
                        ran = True
                    except:
                        ran = False
                time.sleep(3)
            with open(self.embeddings_save_path, "wb") as fOut:
                pickle.dump(doc_embeddings, fOut)
            self.index = faiss.IndexFlatL2(len(doc_embeddings[0]))
            self.index.add(doc_embeddings)
            return doc_embeddings
        else:
            with open(self.embeddings_save_path, "rb") as fIn:
                doc_embeddings = pickle.load(fIn)
                self.index = faiss.IndexFlatL2(len(doc_embeddings[0]))
                self.index.add(doc_embeddings)
                return doc_embeddings
            
    def search(self, query : str, k : int = 5):
        query_embedding = self.get_embed([query])[0]
        distances, indices = self.index.search(np.array([query_embedding]), k)
        return distances, indices
    
    def search_return(self, documents : list[str], query : str, k : int = 5):
        search_results = self.search(query, k)
        return [documents[i] for i in search_results[1].tolist()[0]]
        