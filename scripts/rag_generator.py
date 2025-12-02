 #Load model directly
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain_core.messages import HumanMessage, SystemMessage
import torch
from abc import ABC, abstractmethod

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
BASE_PROMPT = """You are a precise assistant for a retrieval-augmented system. 
              Answer the user's question strictly based on the provided context. 
              Do NOT add any information that is not in the context. 
              If the answer is not present in the context, reply exactly 'I don't know'. 
              Do NOT provide explanations or extra commentary.
              Answer concisely and only with the answer."""


class Generator(ABC):
    def __init__(self, debug=False):
        self.debug = debug

    @abstractmethod
    def run(self, question, evidence):
         pass

    def _log(self, msg):
        if self.debug:
            print(msg)


class Flan_t5(Generator):
     def __init__(self, debug=False):
         super().__init__(debug=debug)
         print("Loading google/flan-t5-large Generator...")

         bnb_config = BitsAndBytesConfig(
             load_in_8bit=True,  # or load_in_4bit=True
             llm_int8_threshold=6.0,
             llm_int8_has_fp16_weight=False,
         )

         flan_t5_large_model_name = "google/flan-t5-small"

         self.tokenizer = AutoTokenizer.from_pretrained(flan_t5_large_model_name)
         self.model = AutoModelForSeq2SeqLM.from_pretrained(
             flan_t5_large_model_name,
             quantization_config=bnb_config,  # OR load_in_4bit=True (requires bitsandbytes >= 0.39)
         )
         print("google/flan-t5-large Generator loaded.")

     def flan_t5_large(self, question, evidence):
         # Example prompt
         prompt = BASE_PROMPT + "\nContext: " + evidence + "Question: " + question

         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
         outputs = self.model.generate(**inputs, max_new_tokens=100)

         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

     def run(self, question, evidence):
         self._log("Generator loading...")
         torch.cuda.empty_cache()
         answer = self.flan_t5_large(question, evidence)
         torch.cuda.empty_cache()
         self._log("Generator done.")
         return answer


class Longformer(Generator):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        print("Loading longform Generator...")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        self.model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        print("Longform Generator Loaded...")

    def longformer(self, question, evidence):
        encoding = self.tokenizer(question, evidence, return_tensors="pt")
        input_ids = encoding["input_ids"]

        # default is local attention everywhere
        # the forward method will automatically set global attention on question tokens

        attention_mask = encoding["attention_mask"]
        outputs = self.model(input_ids, attention_mask=attention_mask)

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        answer_tokens = all_tokens[torch.argmax(start_logits): torch.argmax(end_logits) + 1]
        answer = self.tokenizer.decode(
        self.tokenizer.convert_tokens_to_ids(answer_tokens)
        )
        return answer

    def run(self, question, evidence):
        self._log("Generator loading...")
        answer = self.longformer(question, evidence)
        self._log("Generator done.")
        return answer


class Five_nano(Generator):
     # Necessary OPENAI_API_KEY set up as an enviroment variable
     def __init__(self, debug=False):
         super().__init__(debug=debug)
         print("Loading ChatGPT-5 Nano Generator...")
         self.client = OpenAI()
         self.model_name = "gpt-5-nano"
         print("ChatGPT-5 Nano Generator Loaded.")

     def run(self, question, evidence):
         self._log("Generator loading...")
         try:
             response = self.client.chat.completions.create(
                 model=self.model_name,
                 messages=[
                     {"role": "system", "content": BASE_PROMPT},
                     {"role": "user", "content": (f"Context: {evidence}\n\n"
                                                  f"Question: {question}")}
                 ],
                 max_completion_tokens=1000,  # short limit to avoid extra text
                 top_p=1.0,  # use full probability distribution
                 n=1  # one response only
             )
             self._log("Generator done.")
             return response.choices[0].message.content

         except Exception as e:
             return f"Error connecting to ChatGPT-5 Nano: {e}"
        