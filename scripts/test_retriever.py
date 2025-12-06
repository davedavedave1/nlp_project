# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from rag_generator import Generator, Flan_t5, Longformer
from retriever import Retriever
from reranker import Reranker
import json
from rag_system import Rag_system
import string
import re



class Retriever_with_reranker:
    def __init__(self, path_to_db, how_many_docs, how_many_docs_after_reranker, use_reranker, debug=False):
        print("Loading RAG-System...")
        if not use_reranker:
            how_many_docs=how_many_docs_after_reranker
        self.use_reranker=use_reranker
        self.debug = debug
        self.retr = Retriever(path_to_db, how_many_docs)
        self.reranker = Reranker()
        self.how_many_docs_after_reranker= how_many_docs_after_reranker
        print("RAG-System loaded...")

    def _log(self, msg):
        if self.debug:
            print(msg)

    def run(self, question):
        self._log("Getting Evidence...")
        evidence = self.retr.run(question)

        if self.use_reranker:
            evidence = self.reranker.run(evidence, question,self.how_many_docs_after_reranker)

        
        
        return evidence
    


def test_retriever(file_path, path_to_db, how_many_docs,how_many_docs_after_reranker, use_reranker, debug=False):
    # Open and load the JSON file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    searches_done = with_answer_doc = without_answer_doc = 0
    ret = Retriever_with_reranker(path_to_db,how_many_docs,how_many_docs_after_reranker,use_reranker, debug=debug)

    for i, item in enumerate(data["Data"][:1000]):
        if debug:
            print(item["Question"])

        docs_with_solution=item["EntityPages"]

        retrieved_chunks= ret.run(item["Question"])

        retrieved_right_chunk=False

        for retrieved_chunk in retrieved_chunks:
            retrieved_filename=retrieved_chunk.metadata["filename"]
            if any(page["Filename"] == retrieved_filename for page in docs_with_solution):
                retrieved_right_chunk=True

        

        if retrieved_right_chunk:
            with_answer_doc+=1
        else:
            without_answer_doc+=1
        
        searches_done+=1
        

        print("searches done: "+str(searches_done)+" with answer doc: "+str(with_answer_doc))

    return "Used db: "+str(path_to_db)+" How many docs were retrieved: "+str(how_many_docs)+"searches done: "+str(searches_done)+" with answer doc: "+str(with_answer_doc)
    



#just for testing
if __name__ == "__main__":
    testfile="./necessary_parts_triviaqa/wikipedia-development_new.json"
    prefix_path_to_db="./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~"
    suffix_path_to_db="_chunkOverlap~30_WITH_METADATA"
    test_retriever(testfile, prefix_path_to_db + "1024" + suffix_path_to_db,64, 8, True, debug=False)