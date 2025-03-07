import torch, tqdm, inspect, pathlib
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime

from config import configs
config = configs["19M"]

# https://github.com/huggingface/transformers/blob/92c5ca9/examples/pytorch/language-modeling/run_mlm.py
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

preprocessing_num_workers = 32
max_seq_len = config.pop("max_seq_len", 256)
assert max_seq_len < tokenizer.model_max_length
text_column_name = "text"
overwrite_cache = False
max_batch_size = config.pop("max_batch_size", 8)

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
        max_length=max_seq_len + 1,
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
        dataset, batch_size=max_batch_size, shuffle=True)

# TODO: figure out the right way to do this with split=["train", "validation"]
valid = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
valid = valid.map(
    tokenize_function,
    batched=True,
    num_proc=preprocessing_num_workers,
    remove_columns=[text_column_name],
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on dataset line_by_line",
)

valid.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])
valid = torch.utils.data.DataLoader(valid, batch_size=max_batch_size)

from deepseek import *

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
learning_rate = 6e-4
device_type = "cuda"
epochs = 4

class Trainer(Transformer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (Linear, Gate)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, (ParallelEmbedding, Categorical)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, targets=None, mask=None):
        logits = super().forward(tokens)
        if targets is None:
            loss = None
        else:
            if mask is not None:
                logits = logits * mask[..., None]
            loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(
            self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed,
        # otherwise no. i.e. all weight tensors in matmuls + embeddings decay,
        # all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
                f"num decayed parameter tensors: {len(decay_params)}, "
                f"with {num_decay_params:,} parameters")
        print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, "
                f"with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = \
                'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
                optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

@torch.no_grad()
def validation_loss():
    total = 0
    for x in valid:
        logits, loss = model(
                x["input_ids"][:, :-1].to(device_type),
                x["input_ids"][:, 1:].to(device_type),
                x["attention_mask"][:, 1:].to(device_type))
        total += loss
    return total / len(valid)

model = Trainer(ModelArgs(
        max_batch_size=max_batch_size, vocab_size=len(tokenizer),
        max_seq_len=max_seq_len, **config))
model.to(device_type)

optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type)

def ckpt(epoch):
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'config': config,
    }
    torch.save(checkpoint, out_dir / f"ckpt-{epoch}.pt")

loss_ema = 0
loss_ema_alpha = 0.1
out_dir = pathlib.Path(__file__).parents[0] / "jobs"
out_dir /= datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
out_dir.mkdir(parents=True, exist_ok=True)

ckpt(0)
print("validation_loss:", "{:.3f}".format(validation_loss()))
for epoch in range(epochs):
    print(f"epoch {epoch} / {epochs}")
    pbar = tqdm.tqdm(dataset)
    for x in pbar:
        logits, loss = model(
                x["input_ids"][:, :-1].to(device_type),
                x["input_ids"][:, 1:].to(device_type),
                x["attention_mask"][:, 1:].to(device_type))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_ema = loss_ema_alpha * loss.item() + \
                (1 - loss_ema_alpha) * loss_ema
        pbar.set_description("loss - {:.3f}".format(loss_ema))
    ckpt(epoch + 1)
    print("validation_loss:", "{:.3f}".format(validation_loss()))
