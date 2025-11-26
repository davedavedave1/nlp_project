 #Load model directly
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain_core.messages import HumanMessage, SystemMessage
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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


class Mistral:
    def __init__(self):
        print("Loading Mistral Generator...")     
        model_repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

        self.llm = HuggingFaceEndpoint(
            repo_id=model_repo_id,
            task="text-generation",
            max_new_tokens=128,
            temperature=0.1
        )
        self.chat_model = ChatHuggingFace(llm=self.llm)
        print("Mistral Generator Loaded.")

    def run(self, question, evidence):
        print(f"Generator working on: {question}...")
        
        prompt = (
            f"[INST] You are a helpful tour guide. "
            f"Answer the user's question strictly based on the context provided below. "
            f"If the answer is not in the context, simply say 'I don't know'.\n\n"
            f"Context: {evidence}\n\n"
            f"Question: {question} [/INST]"
        )
        try:
            response = self.chat_model.invoke(prompt)
            return response
        except Exception as e:
            return f"Error connecting to Mistral: {e}"

# mistral = Mistral()
# print(mistral.run("What is the capital of Germany?", "France is a country. Its capital is Paris."))
# print(mistral.run("What is the capital of France?", "France is a country. Its capital is Paris."))

class Phi:
    def __init__(self):
        print("Loading Phi Generator...")
        
        repo_id = "microsoft/Phi-3-mini-4k-instruct"

        # --- THE FIX ---
        # We explicitly tell it the URL so it doesn't have to search for it.
        # This prevents the StopIteration error.
        endpoint_url = f"https://router.huggingface.co/models/{repo_id}"

        self.llm = HuggingFaceEndpoint(
            endpoint_url=endpoint_url, # <--- CRITICAL CHANGE
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        )
        print("Phi Generator Loaded.")

    def ask(self, query, evidence):
        # 2. MANUALLY FORMAT THE PROMPT
        # Phi-3 specific format: <|system|>...<|end|><|user|>...<|end|><|assistant|>
        
        prompt = (
            f"<|system|>\n"
            f"You are a tour guide. Answer based strictly on the evidence provided. "
            f"If you don't know, say so.<|end|>\n"
            f"<|user|>\n"
            f"Context: {evidence}\nQuestion: {query}<|end|>\n"
            f"<|assistant|>"
        )

        # 3. Invoke the endpoint directly with the string
        response_text = self.llm.invoke(prompt)
        
        return response_text
    
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


#def generator(question, evidence):
    #phi(question, evidence)
    #return longformer(question, evidence)
    # phi(question, evidence)


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