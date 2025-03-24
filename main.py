from dotenv import dotenv_values
import chromadb
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

config = dotenv_values(".env")
OAI_KEY = config.get("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OAI_KEY

embed_model = OpenAIEmbedding(model='text-embedding-ada-002')


if __name__ == "__main__":
    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("knowledge-base")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # Optional: set LLM if you want RAG-based synthesis
    llm = OpenAI(model="gpt-4o-mini")

    # Query Data from the persisted index
    query_engine = index.as_query_engine()
    context = query_engine.query("What did the author do growing up?")

    index.storage_context.persist(persist_dir="./chroma_db")

    # Query
    query_engine = index.as_query_engine(
        streaming=True,
        similarity_top_k=3
    )

    context = query_engine.query("What are the key features?")
