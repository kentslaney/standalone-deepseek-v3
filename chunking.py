import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    # Remove empty lines
    examples["text"] = [
        line for line in examples["text"]
        if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(examples["text"])

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=32,
    remove_columns=["text"],
    load_from_cache_file=True,
    desc="Running tokenizer on dataset line_by_line",
)

lens = [len(x["input_ids"]) for x in dataset]

def main():
    plt.ecdf(lens)
    plt.xlabel("tokens")
    plt.ylabel("percentile")
    plt.title("wikitext validation set")
    plt.show()

if __name__ == "__main__":
    main()
