 #Load model directly
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain_core.messages import HumanMessage, SystemMessage
import torch


load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")



class Generator:
    def __init__(self):
        
        
        print("Loading Generator...")



        #flan-t5-large

        
        print("Loading google/flan-t5-large Generator...")


        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,      # or load_in_4bit=True
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        flan_t5_large_model_name = "google/flan-t5-small"

        self.flan_t5_large_tokenizer = AutoTokenizer.from_pretrained(flan_t5_large_model_name)
        self.flan_t5_large_model = AutoModelForSeq2SeqLM.from_pretrained(
                                                                            flan_t5_large_model_name,
                                                                            quantization_config=bnb_config,        # OR load_in_4bit=True (requires bitsandbytes >= 0.39)
                                                                        )
        print("google/flan-t5-large Generator loaded.")


        #longformer
        print("Loading longform Generator...")
        
        self.longformer_tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        self.longformer_model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        print("Longform Generator Loaded...")






        
        print("Loading ChatGPT-5 Nano Generator...")

        # Initialize OpenAI client
        self.client = OpenAI()

        # Model name (class = Mistral)
        self.model_name = "gpt-5-nano"

        print("ChatGPT-5 Nano Generator Loaded.")

        print("Generator Loaded.")

    def run(self, question, evidence):
        print("Generator working...")

        # answer = longformer(self.longformer_tokenizer, self.longformer_model, question, evidence)

        # answer=five_nano(self,question,evidence)

        answer = flan_t5_large(self.flan_t5_large_tokenizer, self.flan_t5_large_model, question, evidence)

        print("Generator done...")
        return answer


def five_nano(self, question, evidence):
    try:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise assistant for a retrieval-augmented system. "
                        "Answer the user's question strictly based on the provided context. "
                        "Do NOT add any information that is not in the context. "
                        "If the answer is not present in the context, reply exactly 'I don't know'. "
                        "Do NOT provide explanations or extra commentary. "
                        "Answer concisely and only with the answer."
                        
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Context: {evidence}\n\n"
                        f"Question: {question}"
                    )
                }
            ],
            max_completion_tokens=1000,  # short limit to avoid extra text
            top_p=1.0,        # use full probability distribution
            n=1               # one response only
            )
        print("Generator done")
        print(response.choices[0])
        return response.choices[0].message.content

    except Exception as e:
        return f"Error connecting to ChatGPT-5 Nano: {e}"



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

def flan_t5_large(tokenizer, model, question, evidence):
    # Example prompt
    prompt = """You are a precise assistant for a retrieval-augmented system. 
                Answer the user's question strictly based on the provided context. 
                Do NOT add any information that is not in the context. 
                If the answer is not present in the context, reply exactly 'I don't know'. 
                Do NOT provide explanations or extra commentary.
                Answer concisely and only with the answer.
                Context: """+ evidence + "Question: "+ question
                        

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)




            
        
        
        



        