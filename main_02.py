from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.core.vector_stores import FaissVectorStore  # Corrected import
from llama_index.storage.storage_context import StorageContext
import faiss

documents = SimpleDirectoryReader("data").load_data()

embed_model = OllamaEmbedding(
    model_name="bge-m3:latest",
    api_base="http://localhost:11434",
    api_key="ollama",
)

dimension = 1024
faiss_index = faiss.IndexFlatL2(dimension)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

llm = Ollama(model="qwen2.5:7b")

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
response = query_engine.query("What is the meaning of life?")
print(response)
