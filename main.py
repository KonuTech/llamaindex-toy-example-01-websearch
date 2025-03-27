from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama

embed_model = OllamaEmbedding(
    model_name="bge-m3:latest",
    api_base="http://localhost:11434",
    api_key="ollama",
)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)


llm = Ollama(model="qwen2.5:7b")

query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
response = query_engine.query("What is the meaning of life?")
print(response)
