from datasets import load_dataset



def load_data():
    domains = get_dataset_config_names('trivia_qa')
    print(domains)
    ds = load_dataset('trivia_qa', name="rc.wikipedia")
    simplified_ds = []
    print(ds)
    #for tripel in ds["validation"]
        #simplified_ds
    print(ds["validation"]["question"])
    #print(ds["validation"][0]["question"])
    return ds


if __name__ == "__main__":
    # had to split up the function so we could call the loader function from the retriever
    data = load_data()

    print(data[0]["question"])