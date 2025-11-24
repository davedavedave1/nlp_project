from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class Retriever:
    def __init__(self, path_to_db):
        print("Loading retriever...")
        model_name = path_to_db.split("_")[1].split("~")[1]
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.db = FAISS.load_local(
            path_to_db,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever(search_type="similarity",
                                              search_kwargs={'k': 4})
        print("Retriever loaded.")

    def run(self, question):
        print("Retriever working...")
        evidence = self.retriever.invoke(question)
        print("Retriever done...")
        return evidence




        

# Run the function for testing
if __name__ == "__main__":
    retr = Retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~1024_chunkOverlap~30")
    results = retr.run("What is the capital of France?")