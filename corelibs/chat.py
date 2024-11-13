class Chat:
    system_prompt: str # system prompt
    messages: list[str] # list of messages in this chat

    def __init__(self, message : str = "", system_prompt = "", cloned_chat : "Chat" = None):
        """
        Create a new chat with an optional starting message and an optional system prompt.
        Alternatively, copy from another Chat instance.

        Args:
            message (str): optional starting message for the chat
            system_prompt (str): optional system prompt for the chat
            other (Chat): optional Chat object to copy from
        """
        if cloned_chat is not None:
            # Copy constructor behavior
            self.messages = cloned_chat.messages.copy()
            self.system_prompt = cloned_chat.system_prompt
        else:
            # Standard constructor behavior
            if message == "":
                self.messages = []
            else:
                self.messages = [message]
            self.system_prompt = system_prompt
    
    def appendNew(self, message : str):
        """Append message to the end of the message list, making it the new final message."""
        self.messages.append(message)
    
    def appendContinue(self, message : str):
        """Append message to the end of the final message in the message list."""
        self.messages[-1] += message
        
    def toOAI(self):
        """
        Convert the message list to an OpenAI API compatible list of messages.

        Returns:
            list[dict]: List of messages formatted for OpenAI API.
        """
        oai_messages = []
        if self.system_prompt != "":
            oai_messages.append({"role": "system", "content": self.system_prompt})
        for i, msg in enumerate(self.messages):
            role = "user" if i % 2 == 0 else "assistant"
            oai_messages.append({"role": role, "content": msg})
        return oai_messages
    
    def __str__(self):
        """
        Overload the __str__ method to convert the Chat object to a string representation.

        Returns:
            str: String representation of the Chat object.
        """
        messages = ["System Prompt: " + self.system_prompt]
        for i, message in enumerate(self.messages):
            role = "User" if i % 2 == 0 else "Assistant"
            messages.append(f"{role}: {message}")
        return "\n".join(messages)
