import json

with open('corelibs/RAG_array/textbook.json') as f:
    data = json.load(f)

rag_data = [x['content'] for x in data]

with open('corelibs/RAG_array/rag.json', 'w') as f:
    json.dump(rag_data, f)
