from fastapi import FastAPI
from main_chain import get_main_chain
from load_env import load_keys_to_env

app = FastAPI()
load_keys_to_env()
chatbot = get_main_chain()

@app.get("/query")
def query_llm(question: str): 
    response = chatbot.invoke(question)["response"]
    print(response.content)

    return {"response": response.content}

    
