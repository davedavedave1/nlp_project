from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from get_triviaqa_huggingace import load_data

# def structure_evidence(evidencefolder):


def retriever(question, evidence):
    data = load_data()
    # evidence = structure_evidence(evidencefolder)
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)
    chunked_docs = splitter.split_documents(docs)
    # change docs to evidence 

    # this is the embeddings database
    db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))

    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 4})
    
retriever()
