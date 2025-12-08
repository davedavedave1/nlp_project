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



