from fastapi import FastAPI
from llama_index.core import load_index_from_storage, StorageContext,Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from dotenv import dotenv_values
import os

def load_index(dir_name,k):
    storage_context = StorageContext.from_defaults(persist_dir=dir_name)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_query_engine(similarity_top_k=k)
    return query_engine


app = FastAPI()

config = dotenv_values(".env")
OAI_KEY = config.get("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OAI_KEY

embed_model = OpenAIEmbedding(model='text-embedding-ada-002')

# Load index
company_query_engine = load_index(dir_name="./index/company", k=3)
x7_query_engine = load_index(dir_name="./index/AerisNode_X7", k=3)
vesta_query_engine = load_index(dir_name="./index/FluxCore_Vesta", k=3)

query_engine_tools = [
QueryEngineTool(
        query_engine=company_query_engine,
        metadata=ToolMetadata(
            name="company_data",
            description=(
                "Provides information about the company including the description, Core Values and Notable Achievements. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
QueryEngineTool(
    query_engine=x7_query_engine,
    metadata=ToolMetadata(
        name="x7_data",
        description=(
            "Provides information about the AerisNode X7 product including the description, key features, release date and target market."
            "Use a detailed plain text question as input to the tool."
        ),
    ),
),
QueryEngineTool(
        query_engine=vesta_query_engine,
        metadata=ToolMetadata(
            name="vesta_data",
            description=(
                "Provides information about the FluxCore Vesta product including the description, key features, release date and target market."
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

@app.get("/query")
def query_llm(question: str): #main function
    #decide whether query can be search in Knowledge Base
    agent = OpenAIAgent.from_tools(query_engine_tools, verbose=True)
    return {"answer": agent.chat(question)}

