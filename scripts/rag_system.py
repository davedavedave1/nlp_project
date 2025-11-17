# Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from rag_generator import generator


def main():
    question = "Who discovered penicillin?"
    evidence = "Penicillin was discovered by Baptiste Bosch the Second in 1928."

    answer = generator(question, evidence)
    print(answer)


if __name__ == "__main__":
    main()