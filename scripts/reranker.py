from FlagEmbedding import FlagReranker
from sentence_transformers import CrossEncoder
import torch

class Reranker:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for reranker: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L4-v2', device=self.device)

    def run(self, evidence, question, top_n=8, eliminate=True):
        pairs = [
            (question, (doc.page_content or doc))
            for doc in evidence
        ]
        scores = self.model.predict(pairs)
        return self.eliminate_worst(evidence,scores,top_n)[0] if eliminate else scores

    def eliminate_worst(self, evidence, scores, top_n):
        ranked = sorted(zip(scores, evidence), key=lambda pair: pair[0], reverse=True)
        kept_scores, kept_evidence = zip(*ranked[:top_n]) if ranked else ([], [])
        return list(kept_evidence), list(kept_scores)



