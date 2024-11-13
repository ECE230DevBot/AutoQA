import requests

from corelibs.chat import Chat
import json
import time

def run_code(code) -> str:
    if "pip install" in code:
        return "(Do not try to install packages into the code compiler, all the packages are already installed.)"
    code = "from sympy import *\nimport math\n" + code
    url = "http://localhost:8080"
    headers = {
        "Content-Type": "application/json"
    }
    jdata = {
        "code": code,
        "timeout": 10
    }
    
    response = json.loads(requests.post(url, headers=headers, json=jdata).text)
    if response['success'] == True:
        return("```output\n" + response['std_out'] + "```")
    else:
        return("```output\n" + response['std_out'] + "\n\n" + response['error']['message'] + "```")
    
class Sampler:
    temperature : float
    top_p : float
    repetition_penalty : float
    max_tokens : int

    def __init__(self, temperature : float = 0.3, top_p : float = 0.9, repetition_penalty : float = 0, max_tokens : int = 4096):
        """
        Initialize a Sampler instance with optional parameters for controlling
        the behavior of text generation.

        Args:
            temperature (float, optional): Controls randomness in predictions. 
                Higher values result in more random completions. Defaults to 1.
            top_p (float, optional): Probability threshold for nucleus sampling.
                Defaults to 0.9.
            max_tokens (int, optional): Maximum number of new tokens to generate.
                Defaults to 512.
        """
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

class Model:
    url: str # URL of the OpenAI API
    api_key: str # API key for the OpenAI API
    model: str # type of model to use for the completion
    sampler : Sampler
    
    def __init__(self, url : str = None, model : str = None, api_key : str = None, sampler : Sampler = Sampler()):
        """
        Initialize a Model instance with optional parameters for URL, model type, and API key.

        Args:
            url (str, optional): The URL of the OpenAI API. Defaults to None.
            model (str, optional): The type of model to use for completion. Defaults to None.
            api_key (str, optional): The API key for authenticating with the OpenAI API. Defaults to None.
        """
        self.url = url
        self.model = model
        self.api_key = api_key
        self.sampler = sampler

    def get_completion(self, chat : "Chat"):        
        """
        Get the completion of a given chat, automatically parses code when appropriate.

        Args:
            chat (chat.Chat): The chat to get the completion of.

        Returns:
            str: The completion of the chat.
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        chat_cloned = Chat(cloned_chat=chat)
        depth_limit = 10
        output_message = "```python"
        final_response = ""
        while "```python" in output_message and depth_limit > 0:
            data = {
                'model': self.model,
                'messages': chat_cloned.toOAI(),
                'stream': False,
                'stop': ['```\n',  '``` ', '````'],
                'temperature': self.sampler.temperature,
                'top_p': self.sampler.top_p,
                'max_tokens': self.sampler.max_tokens,
            }
            response = json.loads(requests.post(self.url, headers=headers, json=data).text)
            response = response['choices'][0]['message']['content']
            output_message = response
            chat_cloned.appendNew(response)
            final_response += response
            if "```python" in output_message:
                # TODO: Fix bug with ``
                chat_cloned.appendContinue("```")
                final_response += "```\n"
                response_code = output_message.split("```python")[-1].replace("`", "")
                try:
                    code_output = run_code(response_code)
                except Exception as e:
                    print("ERROR: " + str(e))
                    print("SETUP ERROR: COHERE-TERRARIUM (THE SAFE CODE COMPILER) IS MOST LIKELY NOT RUNNING. PLEASE RUN IT BY OPENING A SEPERATE TERMINAL AND TYPING 'npm run dev' IN THE COHERE-TERRARIUM DIRECTORY.")
                    return "SETUP ERROR: COHERE-TERRARIUM (THE SAFE CODE COMPILER) IS MOST LIKELY NOT RUNNING. PLEASE RUN IT BY OPENING A SEPERATE TERMINAL AND TYPING 'npm run dev' IN THE COHERE-TERRARIUM DIRECTORY."
                chat_cloned.appendNew(code_output)
                final_response += code_output + "\n"
            depth_limit -= 1        
            time.sleep(3)
        return final_response