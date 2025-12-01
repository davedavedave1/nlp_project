from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from load_data import load_data
import torch  # ADD THIS

def embedder(chunk_size, chunk_overlap):
    docs = load_data("necessary_parts_triviaqa/evidence/wikipedia")
    embedding_model_name = 'paraphrase-MiniLM-L3-v2'  # CHANGE: Faster model
    chunk_size = chunk_size#256
    chunk_overlap = chunk_overlap#30
    reduced_size = False
    test_run_string = ""
    
    BATCH_SIZE = 2000  # Process 10k chunks at a time (adjust based on your RAM)
    
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
    
    if reduced_size:
        documents = documents[:10000]
        test_run_string = "_TESTRUN_REDUCED_NUMBER_OF_DOCS"
    
    print(f"Total documents loaded: {len(documents)}")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunked_docs)}")
    
    # ADD: GPU Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #encode_batch_size = 128 if device == 'cuda' else 32

    encode_batch_size = 16   #recommended for my graphicscard ~ David
    
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # CHANGE: Initialize embeddings model with GPU support
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': device},  # ADD THIS
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
            del temp_db  # ADD: Free memory
            print(f"Merged. Total vectors: {db.index.ntotal}")
        
        # ADD: Clear GPU cache
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"Database created with {db.index.ntotal} vectors")
    
    INDEX_PATH = f"databases/FAISS-DB_embeddingModel~{embedding_model_name}_chunkSize~{chunk_size}_chunkOverlap~{chunk_overlap}{test_run_string}"
    print(f"Saving index to {INDEX_PATH}...")
    db.save_local(INDEX_PATH)
    print("Index saved.")

# if __name__ == "__main__":
    #embedder(512,30)
    #embedder(2048,30)
    #embedder(4096,30)
    #embedder(8192,30)
    #embedder(128,30)