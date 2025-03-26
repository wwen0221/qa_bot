
import os
from dotenv import dotenv_values

def load_keys_to_env():
    config = dotenv_values(".env")
    OAI_KEY = config.get("OPENAI_KEY")
    TAVILY_KEY = config.get("TAVILY_KEY")
    os.environ["OPENAI_API_KEY"] = OAI_KEY
    os.environ["TAVILY_API_KEY"] = TAVILY_KEY
