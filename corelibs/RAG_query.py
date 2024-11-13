# Used best (small) model from https://huggingface.co/spaces/mteb/leaderboard
# I have no idea if this works or not
from sentence_transformers import SentenceTransformer
import pickle
import os
import numpy as np

# ！The default dimension is 1024, if you need other dimensions, please clone the model and modify `modules.json` to replace `2_Dense_1024` with another dimension, e.g. `2_Dense_256` or `2_Dense_8192` !
model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)
query_prompt_name = "s2p_query"

embedding_cache_path = "corelibs/embeddings/"

def get_query_embeddings(queries):
    query_embeddings = model.encode(queries, show_progress_bar=True)
    return query_embeddings

def get_doc_embeddings(docs):
    if not os.path.exists(embedding_cache_path):
        doc_embeddings = model.encode(docs, show_progress_bar=True)
        with open(embedding_cache_path, "wb") as fOut:
            pickle.dump({'sentences': docs, 'embeddings': doc_embeddings}, fOut)
        return doc_embeddings
    else:
        with open(embedding_cache_path, "rb") as fIn:
            cache_data = pickle.load(fIn)
            doc_embeddings = cache_data['embeddings']
            return doc_embeddings
        
def get_similarity(query_embeds, doc_embeds):
    return model.similarity(query_embeds, doc_embeds)

# This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
# They are defined in `config_sentence_transformers.json`

queries = [
    "What are some ways to reduce stress?",
    "What are the benefits of drinking green tea?",
]
docs = [
    "There are many effective ways to reduce stress. Some common techniques include deep breathing, meditation, and physical activity. Engaging in hobbies, spending time in nature, and connecting with loved ones can also help alleviate stress. Additionally, setting boundaries, practicing self-care, and learning to say no can prevent stress from building up.",
    "Green tea has been consumed for centuries and is known for its potential health benefits. It contains antioxidants that may help protect the body against damage caused by free radicals. Regular consumption of green tea has been associated with improved heart health, enhanced cognitive function, and a reduced risk of certain types of cancer. The polyphenols in green tea may also have anti-inflammatory and weight loss properties.",
]
query_embeddings = get_query_embeddings(queries)
doc_embeddings = get_doc_embeddings(docs)
print(query_embeddings.shape, doc_embeddings.shape)
# (2, 1024) (2, 1024)
similarities = get_similarity(query_embeddings, doc_embeddings)
print(similarities)
# tensor([[0.8179, 0.2958],
#         [0.3194, 0.7854]])
