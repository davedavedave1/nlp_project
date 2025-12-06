from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L4-v2')

    def run(self, evidence, question, eliminate=True):
        pairs = [
            (question, (doc.page_content or doc))
            for doc in evidence
        ]
        scores = self.model.predict(pairs)
        return self.eliminate_worst(evidence,scores)[0] if eliminate else scores

    def eliminate_worst(self, evidence, scores, top_n = 4):
        ranked = sorted(zip(scores, evidence), key=lambda pair: pair[0], reverse=True)
        kept_scores, kept_evidence = zip(*ranked[:top_n]) if ranked else ([], [])
        return list(kept_evidence), list(kept_scores)



# _____________________________________
# print("starting reranker 1")
#
# scores = model.predict([
#     ("How many people live in Berlin?", "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."),
#     ("How many people live in Berlin?", "Berlin is well known for its museums."),
# ])
# print("model 1 scores:", scores)

# reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
#
# query = "What event in 1956 marked the official birth of artificial intelligence as a discipline?"
#
# documents = [
#     "In 1950, Alan Turing published his seminal paper, 'Computing Machinery and Intelligence,' proposing the Turing Test as a criterion of intelligence, a foundational concept in the philosophy and development of artificial intelligence.",
#     "The Dartmouth Conference in 1956 is considered the birthplace of artificial intelligence as a field; here, John McCarthy and others coined the term 'artificial intelligence' and laid out its basic goals.",
#     "In 1951, British mathematician and computer scientist Alan Turing also developed the first program designed to play chess, demonstrating an early example of AI in game strategy.",
#     "The invention of the Logic Theorist by Allen Newell, Herbert A. Simon, and Cliff Shaw in 1955 marked the creation of the first true AI program, which was capable of solving logic problems, akin to proving mathematical theorems."
# ]
# results = reranker.compute_score([[query, document] for document in documents])
# print("model 2 scores:", results)