import torch, tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# https://github.com/huggingface/transformers/blob/92c5ca9/examples/pytorch/language-modeling/run_mlm.py
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

preprocessing_num_workers = 32
max_seq_length = 256
assert max_seq_length < tokenizer.model_max_length
text_column_name = "text"
overwrite_cache = False
batch_size = 32

def tokenize_function(examples):
    # Remove empty lines
    examples[text_column_name] = [
        line for line in examples[text_column_name]
        if len(line) > 0 and not line.isspace()
    ]
    return tokenizer(
        examples[text_column_name],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length + 1,
    )

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=[text_column_name],
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset line_by_line",
)
# EOF run_mlm.py

dataset.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
dataset = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

from nanogpt import GPT, GPTConfig

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
learning_rate = 6e-4
device_type = "cuda"
epochs = 1000

model = GPT(GPTConfig(vocab_size=len(tokenizer), n_embd=144))
model.to(device_type)

optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type)

for epoch in range(epochs):
    print(f"epoch {epoch} / {epochs}")
    for x in tqdm.tqdm(dataset):
        logits, loss = model(
                x["input_ids"][:, :-1].to(device_type),
                x["input_ids"][:, 1:].to(device_type),
                x["attention_mask"][:, 1:].to(device_type))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
