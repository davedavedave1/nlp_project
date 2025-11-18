from datasets import load_dataset

def load_data():
    print("Loading dataset...")
    ds = load_dataset('trivia_qa', name="rc.wikipedia", split="validation")    
    return ds

if __name__ == "__main__":
    # had to split up the function so we could call the loader function from the retriever
    data = load_data()

    print(data[0]["question"])