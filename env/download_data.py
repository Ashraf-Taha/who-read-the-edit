"""Fetch the public datasets used in the experiments into the local Hugging Face cache."""
from datasets import load_dataset

DATASETS = [
    ("Salesforce/wikitext", "wikitext-103-raw-v1", "validation"),
    ("hotpotqa/hotpot_qa", "distractor", "validation"),
]

if __name__ == "__main__":
    for path, name, split in DATASETS:
        data = load_dataset(path, name, split=split)
        print(path, name, split, "->", len(data), "rows;", data.column_names)
