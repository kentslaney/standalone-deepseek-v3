import torch, tqdm, inspect, pathlib
from datasets import load_dataset
from transformers import AutoTokenizer
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
from deepseek import *

# https://github.com/huggingface/transformers/blob/92c5ca9/examples/pytorch/language-modeling/run_mlm.py
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_wrapper(max_seq_len):
    assert max_seq_len < tokenizer.model_max_length

    def tokenize_function(examples):
        # Remove empty lines
        examples["text"] = [
            line for line in examples["text"]
            if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_len + 1,
        )
    return tokenize_function

def preprocess(split, max_seq_len, batch_size, shuffle, num_proc=32):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    dataset = dataset.map(
        tokenize_wrapper(max_seq_len),
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset line_by_line",
    )

    dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask"])
    return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle)

class Trainer(Transformer):
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    learning_rate = 6e-4

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
            self, device_type, weight_decay=None, learning_rate=None,
            betas=None):
        weight_decay = self.weight_decay if weight_decay is None else \
                weight_decay
        learning_rate = self.weight_decay if learning_rate is None else \
                learning_rate
        betas = (self.beta1, self.beta2) if betas is None else betas

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

def main(config, epochs=4, resume=None, device_type="cuda"):
    max_seq_len = config.pop("max_seq_len", 256)
    max_batch_size = config.pop("max_batch_size", 8)

    dataset = preprocess("train", max_seq_len, max_batch_size, True)
    valid = preprocess("validation", max_seq_len, max_batch_size, False)

    model = Trainer(ModelArgs(
            max_batch_size=max_batch_size, vocab_size=len(tokenizer),
            max_seq_len=max_seq_len, **config))
    if resume is not None:
        model.load_state_dict(resume["model"])
    model.to(device_type)

    optimizer = model.configure_optimizers(device_type)
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])

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

    first_epoch = 0 if resume is None else resume['epoch']
    if first_epoch != epochs:
        ckpt(first_epoch)
    print("validation_loss:", "{:.3f}".format(validation_loss()))
    for epoch in range(first_epoch, epochs):
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

if __name__ == "__main__":
    from config import configs
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", default=None)
    group.add_argument("--ckpt", default=None)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        main(ckpt["config"], args.epochs, ckpt)
    else:
        config = configs.values()[0] if args.preset is None else \
                configs[args.preset]
        main(config, args.epochs)
