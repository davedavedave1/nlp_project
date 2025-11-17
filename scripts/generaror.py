 #Load model directly
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


def generator(question, evidence):
    tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
    model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

    question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    encoding = tokenizer(question, text, return_tensors="pt")

    input_ids = encoding["input_ids"]

    # default is local attention everywhere

    # the forward method will automatically set global attention on question tokens

    attention_mask = encoding["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)

    start_logits = outputs.start_logits

    end_logits = outputs.end_logits

    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    answer_tokens = all_tokens[torch.argmax(start_logits) : torch.argmax(end_logits) + 1]

    answer = tokenizer.decode(

        tokenizer.convert_tokens_to_ids(answer_tokens)

    )  # remove space prepending space token
    print(answer)

