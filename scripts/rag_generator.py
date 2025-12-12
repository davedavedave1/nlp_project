 #Load model directly
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from langchain_core.messages import HumanMessage, SystemMessage
import torch
from abc import ABC, abstractmethod
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

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


class Llama3_HighSpeed(Generator):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        print("Loading High-Speed Llama-3-8B Generator...")

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN not set")

        # Check GPU memory to decide config
        # L4 has ~22GB usable, A100 has 40GB+
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if self.device == 'cuda' else 0

        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        if gpu_mem > 23:  # A100 or L4 (usually shows as ~22-24GB)
            print(f"High-Performance GPU detected ({gpu_mem:.1f} GB). Loading in native bfloat16...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,  # Native precision = FASTEST
                device_map="auto",
                token=hf_token
            )
        else:
            print(f"Smaller GPU detected ({gpu_mem:.1f} GB). Loading in 4-bit...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                token=hf_token
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Generator loaded.")


    def run(self, question, evidence):
         self._log("Generator (Llama 3) loading...")

         # 3. Apply Llama 3 specific chat template
         messages = [
             {"role": "system", "content": BASE_PROMPT},
             {"role": "user", "content": f"Context: {evidence}\n\nQuestion: {question}"}
         ]

         input_ids = self.tokenizer.apply_chat_template(
             messages,
             add_generation_prompt=True,
             return_tensors="pt"
         ).to(self.model.device)

         # 4. Generate
         terminators = [
             self.tokenizer.eos_token_id,
             self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
         ]

         with torch.no_grad():
             outputs = self.model.generate(
                 input_ids,
                 max_new_tokens=256,
                 eos_token_id=terminators,
                 do_sample=True,  # Set False for deterministic results
                 temperature=0.1,
                 top_p=0.9,
             )

         # Slice off the input prompt to get just the answer
         response = outputs[0][input_ids.shape[-1]:]
         answer = self.tokenizer.decode(response, skip_special_tokens=True)

         self._log("Generator done.")
         return answer


class Flan_t5(Generator):
     def __init__(self, debug=False):
         super().__init__(debug=debug)

         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
         print(f"Using device: {self.device}")
         if self.device == 'cuda':
             print(f"GPU: {torch.cuda.get_device_name(0)}")

         model_kwargs = {}
         if self.device == 'cuda':
             model_kwargs["quantization_config"] = BitsAndBytesConfig(
                 load_in_8bit=True,  # or load_in_4bit=True
                 llm_int8_threshold=6.0,
                 llm_int8_has_fp16_weight=False,
             )
             model_kwargs["device_map"] = "auto"

         flan_t5_large_model_name = "google/flan-t5-small"

         self.tokenizer = AutoTokenizer.from_pretrained(flan_t5_large_model_name)
         self.model = AutoModelForSeq2SeqLM.from_pretrained(
             flan_t5_large_model_name,
             **model_kwargs
         )
         if self.device != 'cuda':
             self.model.to(self.device)
         print("google/flan-t5-large Generator loaded.")

     def flan_t5_large(self, question, evidence):
         # Example prompt
         prompt = BASE_PROMPT + "\nContext: " + evidence + "Question: " + question

         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
         outputs = self.model.generate(**inputs, max_new_tokens=100)

         return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

     def run(self, question, evidence):
         self._log("Generator loading...")
         if self.device == 'cuda':
             torch.cuda.empty_cache()
         with torch.no_grad():
            answer = self.flan_t5_large(question, evidence)
         if self.device == 'cuda':
             torch.cuda.empty_cache()
         self._log("Generator done.")
         return answer


class Longformer(Generator):
    def __init__(self, debug=False):
        super().__init__(debug=debug)
        print("Loading longform Generator...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        self.model = AutoModelForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
        self.model.to(self.device)
        print("Longform Generator Loaded...")

    def longformer(self, question, evidence):
        encoding = self.tokenizer(question, evidence, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        input_ids = encoding["input_ids"]

        # default is local attention everywhere
        # the forward method will automatically set global attention on question tokens

        outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])

        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        all_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        start_index = torch.argmax(start_logits).item()
        end_index = torch.argmax(end_logits).item()
        answer_tokens = all_tokens[start_index: end_index + 1]
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
        
