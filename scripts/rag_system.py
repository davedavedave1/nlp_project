# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from rag_generator import Generator
from retriever import Retriever


class Rag_system:
    
    def __init__(self):
        print("Loading RAG-System...")
        self.retr = Retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~1024_chunkOverlap~30")
        self.gen = Generator()
        print("RAG-System loaded...")

    def run(self, question):
        evidence = self.retr.run(question)
        evidence_content = [doc.page_content for doc in evidence]
        evidence_concat = " ".join(evidence_content)
        #answer = generator(question, evidence)
        #print("This is the evidence we are working with: "+evidence_concat)

        answer= self.gen.run(question, evidence_concat)
        #print("This is the answer: "+answer)
        return answer


#just for testing
if __name__ == "__main__":
    rag = Rag_system()
    print("Answer: "+rag.run("What is the capital of france?"))