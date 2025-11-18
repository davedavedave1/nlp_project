from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_data import load_data

def retriever(question, evidence):
    docs = load_data("../necessary_parts_triviaqa/evidence/wikipedia")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(docs)
    
    # this is the embeddings database
    db = FAISS.from_documents(
        chunked_docs, 
        HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    )
    
    retriever_obj = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4}
    )
    
    