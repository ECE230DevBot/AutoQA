import streamlit as st
from corelibs.model import Model
from corelibs.chat import Chat
from corelibs.embedding_model import Embedding_Model
import json
import os
from datetime import datetime
import requests
from requests.exceptions import RequestException

# Load API key from file
try:
    with open('api_key.txt') as f:
        default_api_key = f.read().strip()
except FileNotFoundError:
    default_api_key = ""

# Load default prompt
try:
    with open('example_prompt.txt') as f:
        default_prompt = f.read().strip()
except FileNotFoundError:
    default_prompt = "Please provide a prompt template with {{question}} and {{context}} placeholders"

# Load RAG data
try:
    with open('corelibs/RAG_array/rag.json') as f:
        rag_data = json.load(f)
except FileNotFoundError:
    rag_data = []

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None
if 'rag_data' not in st.session_state:
    st.session_state.rag_data = None

# Initialize default configuration values
if 'initialized' not in st.session_state:
    st.session_state.main_url = "https://api.openai.com/v1/chat/completions"
    st.session_state.main_model = "gpt-4o-mini"
    st.session_state.main_api_key = default_api_key
    st.session_state.embed_url = "https://api.openai.com/v1/embeddings"
    st.session_state.embed_model = "text-embedding-3-large"
    st.session_state.embed_api_key = default_api_key
    st.session_state.prompt_template_value = default_prompt
    st.session_state.initialized = True
    
def save_configuration():
    try:
        st.session_state.model = Model(url=st.session_state.main_url, 
                                     model=st.session_state.main_model, 
                                     api_key=st.session_state.main_api_key)
        st.session_state.embeddings_model = Embedding_Model(url=st.session_state.embed_url, 
                                                          api_key=st.session_state.embed_api_key, 
                                                          model=st.session_state.embed_model)
        st.session_state.embeddings_model.setup_doc_embeds(rag_data, override_saves=False)
        st.session_state.prompt_template = st.session_state.prompt_template_value
        st.success("Configuration saved successfully!")
    except Exception as e:
        st.error(f"Error saving configuration: {str(e)}")

def save_prompt_completion(prompt, completion):
    """Save prompt and completion to separate files with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = "qna_cache"
    
    # Save prompt
    prompt_file = os.path.join(cache_dir, f"qna_{timestamp}_prompt.txt")
    with open(prompt_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    
    # Save completion
    completion_file = os.path.join(cache_dir, f"qna_{timestamp}_completion.txt")
    with open(completion_file, 'w', encoding='utf-8') as f:
        f.write(completion)
    return f"qna_{timestamp}"

def check_cohere_terrarium_status():
    try:
        response = requests.get("http://localhost:8080", timeout=2)
        return True
    except RequestException:
        return False

# Save default configuration on first run
if st.session_state.model is None:
    save_configuration()

st.title("AutoQA Interface")

# Create tabs
tab1, tab2 = st.tabs(["Configuration", "Use"])

# Configuration Tab
with tab1:
    st.header("Model Configuration")
    
    # Cohere Terrarium Status
    st.subheader("Cohere Terrarium Status")
    if check_cohere_terrarium_status():
        st.success("✅ Cohere Terrarium server is online!")
    else:
        st.error("❌ Cohere Terrarium server is offline. Please install and/or start the cohere-terrarium server:https://github.com/cohere-ai/cohere-terrarium! Once you have it setup, ignore this message.")
    
    # Main model configuration
    st.subheader("Main Model Settings")
    st.text_input("Main Model URL", key="main_url", on_change=save_configuration)
    st.text_input("Main Model Name", key="main_model", on_change=save_configuration)
    st.text_input("Main Model API Key", type="password", key="main_api_key", on_change=save_configuration)
    
    # Embedding model configuration
    st.subheader("Embedding Model Settings")
    st.text_input("Embedding Model URL", key="embed_url", on_change=save_configuration)
    st.text_input("Embedding Model Name", key="embed_model", on_change=save_configuration)
    st.text_input("Embedding Model API Key", type="password", key="embed_api_key", on_change=save_configuration)
    
    # Prompt configuration
    st.subheader("Prompt Template")
    st.text_area("Edit Prompt Template", height=300, key="prompt_template_value", on_change=save_configuration)

# Use Tab
with tab2:
    st.header("Ask Questions")
    
    if not st.session_state.model or not st.session_state.embeddings_model:
        st.warning("Please configure the models in the Configuration tab first.")
    else:
        # Question input
        question = st.text_area("Enter your question:", height=100)
        
        if st.button("Get Answer"):
            with st.spinner("Processing..."):
                try:
                    # Retrieve relevant context
                    search = st.session_state.embeddings_model.search_return(rag_data, question, 2)
                    context = "\n\n".join(search)
                    
                    # Create final prompt
                    prompt = st.session_state.prompt_template.replace("{{question}}", question).replace("{{context}}", context)
                    # Get AI response
                    chat = Chat(message=prompt)
                    completion = st.session_state.model.get_completion(chat)
                    
                    # Display results
                    st.subheader("Answer:")
                    st.write(completion)
                    
                     # Save the prompt-completion pair
                    try:
                        disp = save_prompt_completion(prompt, completion)
                        st.write(f"Prompt and completion saved to: qna_cache/{disp}_prompt.txt and qna_cache/{disp}_completion.txt")
                    except Exception as e:
                        st.warning(f"Failed to save Q&A pair: {str(e)}")
                    
                    with st.expander("Show Prompt Used"):
                        st.write(prompt.replace("\n", "\n\n"))
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")