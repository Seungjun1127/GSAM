from datasets import load_dataset

def load_word_dataset():
    """
    Load the word dataset.
    """
    # Load the word dataset. For example, the 'sst2' part of the 'glue' dataset
    dataset = load_dataset("glue", "sst2", split="train", trust_remote_code=True)
    
    # Check the size of the dataset.
    print(f"Loaded dataset with {len(dataset)} samples.")
    
    # Print the first 5 samples of the dataset.
    #for i in range(5):
       # print(f"Sample {i + 1}: {dataset[i]['sentence']}")  # Assuming the 'sentence' field contains the sentence.
    
    return dataset


if __name__ == "__main__":
    dataset = load_word_dataset()
    print(dataset)

