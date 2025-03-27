import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.ollama import Ollama
from datasets import load_dataset
from pathlib import Path
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core import SimpleDirectoryReader


def main():
    """
    Main function to create and query a vector store index.
    """
    # Load dataset
    dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")

    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Write persona data to files
    for i, persona in enumerate(dataset):
        with open(data_dir / f"persona_{i}.txt", "w") as f:
            f.write(persona["persona"])

    # Load documents from the data directory
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(len(documents))

    # Initialize embedding model
    embed_model = OllamaEmbedding(
        model_name="qwen2.5:7b",
        api_base="http://localhost:11434",
        api_key="ollama",
    )

    # Create the ingestion pipeline
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            embed_model,
        ]
    )

    # Run the pipeline synchronously
    nodes = pipeline.run(documents=documents[:10])

    # Initialize ChromaDB
    db = chromadb.PersistentClient(path="./alfred_chroma_db")
    chroma_collection = db.get_or_create_collection(name="alfred")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Create the vector store index
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    )

    # Initialize the language model
    llm = Ollama(model="qwen2.5:7b")

    # Create the query engine
    query_engine = index.as_query_engine(
        llm=llm,
        response_mode="tree_summarize",
    )

    # Query the engine
    response = query_engine.query(
        "Respond using a persona that describes author and travel experiences?"
        )
    print(response)


if __name__ == "__main__":
    main()
