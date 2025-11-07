import torch
from bionemo.esm2.data.tokenizer import get_tokenizer

work_dir = "../results"
results = torch.load(f"{work_dir}/predictions__rank_0__dp_rank_0.pt")

for key, val in results.items():
    if val is not None:
        print(f"{key}\t{val.shape}")

#
logits = results["token_logits"].transpose(0, 1)  # s, b, h  -> b, s, h
print(f"Logits shape: {logits.shape}")

#
tokenizer = get_tokenizer()

tokens = tokenizer.all_tokens
print(f"There are {tokenizer.vocab_size} unique tokens: {tokens}.")

aa_logits = logits[..., : tokenizer.vocab_size]  # filter out the 95 paddings and only keep 33 vocab positions
print(f"Logits shape after removing the paddings in hidden dimension: {aa_logits.shape}")

#
aa_tokens = ["L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N", "F", "Y", "M", "H", "W", "C"]

aa_indices = [i for i, token in enumerate(tokens) if token in aa_tokens]
extra_indices = [i for i, token in enumerate(tokens) if token not in aa_tokens]

#
input_ids = results["input_ids"]  # b, s
# mask where non-amino acid tokens are True
mask = torch.isin(input_ids, torch.tensor(extra_indices))
