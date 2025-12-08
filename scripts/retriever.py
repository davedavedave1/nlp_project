from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class Retriever:
    def __init__(self, path_to_db, how_many_docs):
        print("Loading retriever...")
        model_name = path_to_db.split("_")[1].split("~")[1]
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device}
        )

        self.db = FAISS.load_local(
            path_to_db,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.retriever = self.db.as_retriever(search_type="similarity",
                                              search_kwargs={'k': how_many_docs})


        print("Retriever loaded.")

        comment="""
        print("Loading google/flan-t5 Generator for the Retriever...")


        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,      # or load_in_4bit=True
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )

        flan_t5_model_name = "google/flan-t5-small"

        self.flan_t5_tokenizer = AutoTokenizer.from_pretrained(flan_t5_model_name)
        self.flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(
                                                                            flan_t5_model_name,
                                                                            quantization_config=bnb_config,        # OR load_in_4bit=True (requires bitsandbytes >= 0.39)
                                                                        )
        print("google/flan-t5 for the Retriever loaded.")


        
        print("Loading Qwen2.5-1.5B Generator for the Retriever...")

        # 8-bit or 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,                # Qwen works great in 4-bit
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )

        qwen_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        # Tokenizer + Model
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(
            qwen_model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )

        print("Qwen2.5-1.5B for the Retriever loaded.")"""


    def run(self, question):
        #print("Retriever working...")
        #example_evidence= generate_example_evidence_flan_t5(self.flan_t5_tokenizer, self.flan_t5_model, question)

        evidence = self.retriever.invoke(question)
        #print("Retriever done...")
        return evidence


prompt1="""Write a long, detailed text (about 1000 tokens) about the topic of the QUESTION.
The text should look like an informative article that could contain an answer.
Do NOT answer the question. Do NOT mention the question. 
Do NOT explain what you are doing. Just write the text.

QUESTION: """

prompt2="""Write a ~1000-token document about the topic of the QUESTION. 
It should read like real informative text that could contain the answer, 
but must NOT answer the question. No meta comments, no headings. 
Output only the document.

QUESTION: """

prompt3="""You are generating an "example evidence document" for a RAG system.

Given a QUESTION, produce a long text (~1000 tokens) that:
- is on the same topic as the question,
- could realistically be a context passage containing the answer,
- is detailed, factual, and coherent,
- does NOT explicitly answer the question,
- does NOT mention that it is synthetic or an example,
- does NOT include metadata, titles, or section headings.

Output ONLY the document text.

QUESTION: """

prompt4="""Write a long, detailed article, about 1000 tokens. 
It must be about the topic of the QUESTION. 
Do NOT answer the question. Just write the article.

QUESTION: """

def generate_example_evidence_flan_t5(tokenizer, model, question):
    # Example prompt
    
                        
    prompt= prompt1+question
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, 
                            max_length=300,          # allow ~300 tokens output
                            min_length=200,           # enforce long output
                            temperature=0.85,         # mild creativity but still stable
                            top_p=0.92,               # controlled sampling
                            top_k=50,                 # avoids degenerate tokens
                            repetition_penalty=1.25,  # prevents loops & restatement
                            no_repeat_ngram_size=5,   # crucial for long documents
                            num_beams=1,              # sampling only (beams produce too-short, safe output)
                            do_sample=True,           # required for variation & length
                            length_penalty=1.1        # encourages slightly longer sequences
                            )

    print("example evidence doc: "+tokenizer.decode(outputs[0], skip_special_tokens=True))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)




def generate_example_evidence_Qwen(tokenizer, model, question):
    prompt = prompt1 + question

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Qwen is decoder-only, so we use "generate" normally
    outputs = model.generate(
        **inputs,
        max_new_tokens=1500,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.05
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("example evidence doc:", text)
    return text


        

# Run the function for testing
if __name__ == "__main__":
    retr = Retriever("./databases/FAISS-DB_embeddingModel~paraphrase-MiniLM-L3-v2_chunkSize~1024_chunkOverlap~30", 2)
    results = retr.run("What is the capital of France?")
    print(results)
