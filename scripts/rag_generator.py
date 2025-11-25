 #Load model directly
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain_core.messages import HumanMessage, SystemMessage
import torch

# Might be more "best-practice" if we would implement this function in the Generator class as well
# I think it would add more structure to our code but not I'm not sure about it
def longformer(tokenizer, model, question, evidence):
    

    #print("Lonngformer got this Question and Evidence: ("+question+","+evidence+")")

    encoding = tokenizer(question, evidence, return_tensors="pt")

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
    return answer

def phi(question, evidence):
    llm = HuggingFaceEndpoint(repo_id = "microsoft/Phi-3.5-mini-instruct", task = "text-generation")
    chat_model = ChatHuggingFace(llm = llm)
    query = question
    response = ask(query, evidence, chat_model)
    print(response)


#def generator(question, evidence):
    #phi(question, evidence)
    #return longformer(question, evidence)
    # phi(question, evidence)

class Phi:
    def __init__(self):
        print("Loading Phi Generator...")
        llm = HuggingFaceEndpoint(repo_id = "microsoft/Phi-3.5-mini-instruct", task = "text-generation")
        self.model = ChatHuggingFace(llm = llm)
        print("Phi Fenerator Loaded...")

    # Define a function to ask the LLM
    def ask(self, query, evidence):
        messages = [
            SystemMessage(content = "You are a tour guide"),
            HumanMessage(content = f"Answer the {query} based on the {evidence}")
        ]
        response = self.model.invoke(messages)
        return response
    
    def run(self, question, evidence):
        

class Generator:
    def __init__(self):
        print("Loading longform Generator...")
        #longformer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        self.model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        print("Longform Generator Loaded...")

    def run(self, question, evidence):
        return longformer(self.tokenizer, self.model, question, evidence)

