from corelibs.model import Model
from corelibs.chat import Chat
from corelibs.embedding_model import Embedding_Model
import json
import time

print("Initializing...")
# Get API key to use models
with open('api_key.txt') as f:
    api_key = f.read().strip()
# Example system prompt to encourage step by step thinking and code execution
with open('example_prompt.txt') as f:
    example_prompt = f.read().strip()
# RAG data for reference
with open('corelibs/RAG_array/rag.json') as f:
    rag_data = json.load(f)

## SETUP
model = Model(url="https://api.openai.com/v1/chat/completions", model="gpt-4o-mini", api_key=api_key)
embeddings_model = Embedding_Model(url='https://api.openai.com/v1/embeddings', api_key=api_key, model="text-embedding-3-large")
embeddings_model.setup_doc_embeds(rag_data, override_saves=False)


# Type whatever question you want
questions = [
            """Building off of our discussion in lecture about the "particle in an infinite potential well," consider a particle in a finite potential well between 00 and aa in 1-dimension. The potential energy is finite (V0)(V0​) outside of the box and 00 within the box. Beginning with the time-independent Schrödinger equation, complete the following parts: (30 points)

a) Set up the set of wave function equations for the three distinct regions in xx (these can include unknown constants).

b) What are the boundary conditions (hint: wave function and its first derivative must be continuous)?

c) Applying the boundary conditions, determine the expression that describes how the wave number is constrained in order to yield solutions to the wave equation (hint: the answer should be an equation in the form tan⁡(ka)=(some function of k,E, etc.)tan(ka)=(some function of k,E, etc.)).

d) Conceptualization: Disregarding the equations and analysis from parts a-c), sketch the wave function at the first three energy levels within this finite potential well and on the same plot, sketch the wave functions for if the well were an infinite potential well. How are the wave functions qualitatively different?""",
            """(a) The wavelength of green light is λ = 550 nm. If an electron has the same wave-length, determine the electron velocity and momentum. (b) Repeat part (a) for the light with a wavelength of λ = 440 nm. (c) For parts (a) and (b), is the photon equal to the momentum of the electron?""",
            """(a) The de Broglie wavelength of an electron is 85 Å. Determine the electron energy (eV), momentum, and velocity. (b) An electron is moving with a velocity of 8 x 10^5 cm/s. Determine the electron energy (eV), momentum, and de Broglie wavelength (in Å).""",
            """An electron is described by a wave function given by ψ(x) = √(2/a) cos(πx/a) for -a/2 < x < a/2. The wave function is zero elsewhere. Calculate the probability of finding the electron between (a) 0 < x < a/4, (b) a/4 < x < a/2, and (c) -a/2 < x < a/2.""",
            """(a) An electron is bound in a one-dimensional infinite potential well with a width of 10 Å. (a) Calculate the first three energy levels that the electron may occupy. (b) If the electron drops from the third to the second energy level, what is the wavelength of a photon that might be emitted?""",
            ]

for i in range(len(questions)):
    question = questions[i]
    # Retrieve top 2 most relevant queries
    search = embeddings_model.search_return(rag_data, question, 2)
    # Join queries
    context = "\n\n".join(search)         
    # Create final prompt
    prompt = example_prompt.replace("{{question}}", question).replace("{{context}}", context)
    print("Begin question " + str(i + 1))

    # Get AI response
    chat = Chat(message=prompt)
    completion = model.get_completion(chat)
    print("End question " + str(i + 1))
    
    # Save the prompt to a text file called qna_results/prompt_N.txt and the completion to a text file called qna_results/completion_N.txt
    with open(f"qna_results/prompt_{i + 1}.txt", "w") as f:
        f.write(prompt)
    with open(f"qna_results/completion_{i + 1}.txt", "w") as f:
        f.write(completion)
print("Finished! Check qna_results/... for results.")
