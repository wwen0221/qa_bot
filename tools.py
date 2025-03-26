
from langchain.tools import tool
from llama_index.core import load_index_from_storage, StorageContext
from langchain_tavily import TavilySearch

def load_index(dir_name,k):
    storage_context = StorageContext.from_defaults(persist_dir=dir_name)
    index = load_index_from_storage(storage_context)
    query_engine = index.as_retriever(similarity_top_k=k)
    return query_engine


@tool
def llamaindex_retrieve_tool(query: str) -> str:
    """Retriever for Apple products, specs and release dates"""
    product_retreiever = load_index(dir_name="./index/products", k=3)
    results = product_retreiever.retrieve(query)
    return "\n\n".join([r.text for r in results])

@tool
def tavily_search_tool(query:str) -> str:
    """Use this tool to search information that are not related to Apple products, specs, and release dates."""
    tool = TavilySearch(
    max_results=3,
    )
    return tool.run(query)
