from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_data import load_data

# run everything from main directory
def embedder():
    docs = load_data("necessary_parts_triviaqa/evidence/wikipedia")
    embedding_model_name='paraphrase-MiniLM-L3-v2'
    chunk_size=512
    chunk_overlap=30
    reduced_size=True

    test_run_string=""
    
    # Convert dictionaries to Document objects
    if docs and isinstance(docs[0], dict):
        documents = [
            Document(
                page_content=doc.get('text', '') or doc.get('content', '') or str(doc),
                metadata=doc.get('metadata', {})
            ) 
            for doc in docs
        ]
    else:
        documents = docs

    #for testruns reduce the number of docs
    if reduced_size:
        documents = documents[:100]
        test_run_string="_TESTRUN_REDUCED_NUMBER_OF_DOCS"
    
    print(f"Total documents loaded: {len(documents)}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(chunked_docs)}")
    
    # this is the embeddings database
    db = FAISS.from_documents(
        chunked_docs, 
        HuggingFaceEmbeddings(model_name=embedding_model_name)
    )
    
    print(f"Database created with {db.index.ntotal} vectors")
    

    INDEX_PATH = "databases/FAISS-DB_embeddingModel~"+embedding_model_name+"_chunkSize~"+str(chunk_size)+"_chunkOverlap~"+str(chunk_overlap)+test_run_string

    print(f"Saving index to {INDEX_PATH}...")
    db.save_local(INDEX_PATH)
    print("Index saved.")

# Run the function
if __name__ == "__main__":
    embedder()