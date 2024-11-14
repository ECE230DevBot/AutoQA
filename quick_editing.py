import json

# Read and split the extra content
with open('extra_content.txt', 'r') as f:
    extra_content = f.read()
    extra_array = [x.strip() for x in extra_content.split('<<SPLITTER>>') if x.strip()]

# Load the original RAG array
with open('corelibs/RAG_array/rag_original.json', 'r') as f:
    original_array = json.load(f)

# Combine both arrays
combined_array = original_array + extra_array

# Save the combined array
with open('corelibs/RAG_array/rag.json', 'w') as f:
    json.dump(combined_array, f)