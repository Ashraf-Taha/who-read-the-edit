"""Fetch the open models used in the experiments into the local Hugging Face cache.

Revisions are pinned to the commits the experiments ran against. The two 7B models are
run in bfloat16 (Section 5.3 and Appendix E); for Mistral only the Hugging Face-format
shards are fetched, not the duplicate consolidated checkpoint.
"""
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen/Qwen2.5-1.5B-Instruct": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": "31b70e2e869a7173562077fd711b654946d38674",
    "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    "mistralai/Mistral-7B-Instruct-v0.3": "c170c708c41dac9275d15a8fff4eca08d52bab71",
}
PATTERNS = ["config.json", "generation_config.json", "model*.safetensors", "model.safetensors.index.json",
            "tokenizer*", "special_tokens_map.json", "vocab.json", "merges.txt"]

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(MODELS)
    for repo in wanted:
        path = snapshot_download(repo, revision=MODELS[repo], allow_patterns=PATTERNS)
        print(repo, "->", path, flush=True)
