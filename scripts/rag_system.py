# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from rag_generator import generator
from retriever import retriever


def rag_system(question):
    
    evidence=retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~512_chunkOverlap~30_TESTRUN_REDUCED_NUMBER_OF_DOCS", question)
    evidence_content = [doc.page_content for doc in evidence]
    evidence_concat = " ".join(evidence_content)
    #answer = generator(question, evidence)

    answer= generator(question, evidence_concat)
    print("This is the answer: "+answer)
    return answer

if __name__ == "__main__":
    rag_system("What is the capital of France?")