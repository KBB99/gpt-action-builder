from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Optional
class CallbackManager(BaseCallbackHandler):
    """Class to manage callback methods for the program."""
    
    def __init__(self):
        """Initialize CallbackManager with empty last_execution list."""
        self.last_execution = []

    def clear(self):
        """Clear the last_execution list."""
        self.last_execution = []
    
    def on_text(self, text: str, **kwargs) -> None:
        """Handle a new token from the Language Learning Model (LLM).

        Args:
            token (str): The new token received from the LLM.
        """
        self.last_execution.append(text)