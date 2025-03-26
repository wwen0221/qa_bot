from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from tools import llamaindex_retrieve_tool, tavily_search_tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from memory import load_memory, load_chat_history, save_to_memory
from operator import itemgetter

def function_calling(tool_call):
    try:
        name = tool_call["name"]
        args = tool_call["arguments"]
        if name == "llamaindex_retrieve_tool":
            return llamaindex_retrieve_tool.run(args)
        elif name == "tavily_search_tool":
            return tavily_search_tool.run(args)
        return "Unknown tool"
    except:
        return "Unknown tool"


def debug_step(tag):
    return RunnableLambda(lambda x: (print(f"[DEBUG - {tag}]: {x}") or x))

def extract_and_call_tool(data):
    tool_call = data["action"].additional_kwargs.get("function_call", {})
    tool_result = function_calling(tool_call)
    return {"question": data["question"], "tool_result": tool_result}

def get_main_chain():
    llm = ChatOpenAI(model = 'gpt-4o-mini', temperature=0)
    tool_prompt = PromptTemplate.from_template("""
    
    Conversation so far:
    {chat_history}

    The user asked: {question}
    If you need to call a tool, respond with a function_call JSON.
    """)

    prompt = PromptTemplate.from_template("""
    You are a helpful assistant that can answer questions about Apple products and their features.
    You should not answer any query that is unrelated to Apple, even though you can.
    Conversation so far:
    {chat_history}

    User's query: {question} 
    Answer the user's query by refering to the following tools result (if any):
    {tool_result}
    """)

    tools = [llamaindex_retrieve_tool, tavily_search_tool]
    functions = [convert_to_openai_function(t) for t in tools]
    memory = load_memory()

    final_output = {
        'response': prompt | llm,
        'question': itemgetter('question'),
    }

    query_chain = (
        RunnablePassthrough()
        | debug_step("query_received")
        | RunnableLambda(lambda question: {
        "question": question,
        "chat_history": load_chat_history(memory).get("chat_history", ""),
        })
        | RunnableLambda(lambda data: {
            "question": data["question"],
            "action": llm.invoke([
                HumanMessage(content=tool_prompt.format(
                    chat_history=data["chat_history"], 
                    question=data["question"]
                ))
            ], functions=functions)
        })
        | debug_step("action_llm_output")
        | RunnableLambda(extract_and_call_tool)
        | RunnableLambda(lambda data: {
            "question": data["question"], 
            "tool_result": data["tool_result"],
            "chat_history": load_chat_history(memory).get("chat_history", "")})
        | debug_step("final_prompt_input")
        | final_output
        | RunnableLambda(lambda response_data: save_to_memory(memory,response_data))
    )
    
    return query_chain