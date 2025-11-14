from datasets import get_dataset_config_names
from datasets import load_dataset



def main():
    #domains = get_dataset_config_names('trivia_qa')
    #print(domains)
    ds = load_dataset('trivia_qa', name="rc.wikipedia")
    simplified_ds = []
    #for tripel in ds["validation"]
        #simplified_ds
    print(ds["validation"]["question"])
    #print(ds["validation"][0]["question"])


if __name__ == "__main__":
    main()