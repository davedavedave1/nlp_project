from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_data import load_data

def retriever(path_to_db, question):
    
    
    embeddings = HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L3-v2')

    # directory where the FAISS index was saved
    db = FAISS.load_local(
        path_to_db,
        embeddings,
        allow_dangerous_deserialization=True  # required in recent LangChain versions
    )
    
    #print(f"Database loaded with {db.index.ntotal} vectors")
    
    retriever_obj = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}# return the 4 most similar docs
    )
    
    #print(f"\nSearching for: '{question}'")

    results = retriever_obj.invoke(question)

    #print(results[0])
    
    return results

        

# Run the function
if __name__ == "__main__":
    results = retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~512_chunkOverlap~30_TESTRUN_REDUCED_NUMBER_OF_DOCS", "What is the capital of France?")