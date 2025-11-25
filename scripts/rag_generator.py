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

# def phi(question, evidence):
#     llm = HuggingFaceEndpoint(repo_id = "microsoft/Phi-3.5-mini-instruct", task = "text-generation")
#     chat_model = ChatHuggingFace(llm = llm)
#     query = question
#     response = ask(query, evidence, chat_model)
#     print(response)

def ministral(question, evidence):
    model_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

    llm = HuggingFaceEndpoint(
        repo_id=model_repo_id,
        task="text-generation",
        max_new_tokens=128,
        temperature=0.1
    )

    chat_model = ChatHuggingFace(llm=llm)
    print(f"Invoking Chat Model: {model_repo_id}")
    query = question

    try:
        response = ask(query, evidence, chat_model)
        # print("RESPONSE: ", response.content)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error communicating with HF API: {e}"


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
            SystemMessage(content = f"You are a tour guide. Just answer the queries based on {evidence}. If you don't have the information, you must say you don't know!"),
            HumanMessage(content = f"Answer the {query} based on the {evidence}")
        ]
        response = self.model.invoke(messages)
        return response
    
    def run(self, question, evidence):
        print("Generator working...")
        answer = self.ask(question, evidence)
        print("Generator done")
        return answer
        
class Generator:
    def __init__(self):
        print("Loading longform Generator...")
        #longformer
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        self.model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        print("Longform Generator Loaded...")

    def run(self, question, evidence):
        print("Generator working...")
        # answer = longformer(self.tokenizer, self.model, question, evidence)
        answer = ministral(question, evidence)
        print("Generator done")
        return answer

