from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_data import load_data

def retriever(path_to_db, question):
    
    model_name=path_to_db.split("_")[1].split("~")[1]#get the name of the model from the path so that we dont have to set it everytime we want to use a different embedding
    #could be crashing if the model_name contains _ or ~
    
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

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

        

# Run the function for testing
if __name__ == "__main__":
    results = retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~512_chunkOverlap~30_TESTRUN_REDUCED_NUMBER_OF_DOCS", "What is the capital of France?")