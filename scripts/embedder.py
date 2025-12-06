from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_data import load_data
import torch  

def embedder(chunk_size, chunk_overlap, reduced_size):
    
    #Load the evidence data and set the necessary parameters
    docs = load_data("necessary_parts_triviaqa/evidence/wikipedia")
    embedding_model_name = 'paraphrase-MiniLM-L3-v2'  
    BATCH_SIZE = 2000  # Process 2000 chunks at a time (adjust based on your RAM)
    
    # Convert dictionaries to langchain documents
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
    
    if reduced_size: #We can enable reduced_size in case we want to do a testrun and time is limited
        documents = documents[:10000]
        test_run_string = "_TESTRUN_REDUCED_NUMBER_OF_DOCS"
    else:
        test_run_string = ""
    
    print(f"Total documents loaded: {len(documents)}")
    

    #Split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunked_docs = splitter.split_documents(documents)


    # --- Add filename + chunk index to each chunk --- We need this to test the quality of the embedding.
    for i, chunk in enumerate(chunked_docs):
        filename = chunk.metadata.get("filename", "unknown_file")
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_id"] = f"{filename}_chunk_{i}"

    print(f"Total chunks created: {len(chunked_docs)}")

    
    #Make the embedding run on GPU if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encode_batch_size = 16   #suitable setting for small GPU
    
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    
    #Initialize embeddings model 
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': device}, 
        encode_kwargs={  # ADD THIS
            'batch_size': encode_batch_size,
            'normalize_embeddings': True
        }
    )
    
    # Process in batches
    db = None
    total_batches = (len(chunked_docs) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(chunked_docs), BATCH_SIZE):
        batch = chunked_docs[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
        
        if db is None:
            # Create initial database
            db = FAISS.from_documents(batch, embeddings)
        else:
            # Create temporary database and merge
            temp_db = FAISS.from_documents(batch, embeddings)
            db.merge_from(temp_db)
            del temp_db  
            print(f"Merged. Total vectors: {db.index.ntotal}")
        
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"Database created with {db.index.ntotal} vectors")
    
    #create a meaningful name for the db that contains all the necassary information
    INDEX_PATH = f"databases/FAISS-DB_embeddingModel~{embedding_model_name}_chunkSize~{chunk_size}_chunkOverlap~{chunk_overlap}{test_run_string}_WITH_METADATA"
    
    print(f"Saving index to {INDEX_PATH}...")

    #save the db
    db.save_local(INDEX_PATH)

    print("Index saved.")

if __name__ == "__main__":
    #embedder(256,30)
    #embedder(512,30)
    #embedder(1024,30)
    embedder(2048,30, False)
    embedder(4096,30, False)
    embedder(8192,30, False)
    #embedder(128,30)