from langchain.memory import ConversationBufferMemory


def load_memory():
    try:
        return ConversationBufferMemory.load()
    except:
        return ConversationBufferMemory(memory_key="chat_history")
    
def load_chat_history(memory):
    chat_history = memory.load_memory_variables({})['chat_history']
    
    return {"chat_history": chat_history}

def save_to_memory(memory, response_data):
    question = response_data["question"]
    response = response_data["response"]
    # Save both user and AI messages
    memory.save_context({"input":question}, {"output": response.content})
    return response_data