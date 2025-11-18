from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_data import load_data

def retriever(question, evidence):
    docs = load_data("necessary_parts_triviaqa/evidence/wikipedia")
    
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

    documents = documents[:1000]
    
    print(f"Total documents loaded: {len(documents)}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(chunked_docs)}")
    
    # this is the embeddings database
    db = FAISS.from_documents(
        chunked_docs, 
        HuggingFaceEmbeddings(model_name='paraphrase-MiniLM-L3-v2')
    )
    
    print(f"Database created with {db.index.ntotal} vectors")
    
    retriever_obj = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )
    
    # Test search
    test_query = "What is the capital of France?"
    print(f"\nSearching for: '{test_query}'")
    results = retriever_obj.invoke(test_query)
    
    print(f"\nFound {len(results)} results:")
    for i, doc in enumerate(results, 1):
        print(f"\n{'='*60}")
        print(f"Result {i}:")
        print(f"{'='*60}")
        print(doc.page_content[:300])  # Print first 300 characters
        print("...")

        INDEX_PATH = "faiss_index_store"

        print(f"Saving index to {INDEX_PATH}...")
        db.save_local(INDEX_PATH)
        print("Index saved.")
    
    return retriever_obj, db

# Run the function
if __name__ == "__main__":
    retriever_obj, db = retriever(None, None)