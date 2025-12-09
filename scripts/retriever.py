from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class Retriever:
    def __init__(self, path_to_db, how_many_docs):
        print("Loading retriever...")
        model_name = path_to_db.split("_")[1].split("~")[1]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device}
        )

        self.db = FAISS.load_local(
            path_to_db,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever(search_type="similarity",
                                              search_kwargs={'k': how_many_docs})


        print("Retriever loaded.")

        


    def run(self, question):
        #print("Retriever working...")
        if self.device == 'cuda':
             torch.cuda.empty_cache()
        with torch.no_grad():
            evidence = self.retriever.invoke(question)
        if self.device == 'cuda':
             torch.cuda.empty_cache()
        #print("Retriever done...")
        return evidence


        

# Run the function for testing
if __name__ == "__main__":
    retr = Retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~1024_chunkOverlap~30", 2)
    results = retr.run("What is the capital of France?")
    print(results)
