from dotenv import dotenv_values
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

config = dotenv_values(".env")
OAI_KEY = config.get("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OAI_KEY

embed_model = OpenAIEmbedding(model='text-embedding-ada-002')

# Load documents
documents = SimpleDirectoryReader(input_files=["docs/FluxCore Vesta/data.txt"]).load_data()

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model
)

index.storage_context.persist(persist_dir="./index/FluxCore_Vesta")
