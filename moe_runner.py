import torch, tqdm, inspect, pathlib, json, shutil
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

    # TODO: this may cut off paragraphs that should be chunked
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
            return logits
        if mask is not None:
            logits = logits * mask[..., None]
        loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-1)
        if mask is not None:
            loss /= torch.sum(mask)
        else:
            loss /= tokens.shape[0]
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

jobs_dir = pathlib.Path(__file__).parents[0] / "jobs"
loss_filename = "loss.json"

def run(config, epochs=4, resume=None, history=None, device_type="cuda"):
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
            total += loss.item()
        return total / len(valid)

    def ckpt(epoch):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'config': config,
        }
        torch.save(checkpoint, out_dir / f"ckpt-{epoch:04}.pt")

    loss_acc = 0
    loss_acc_steps = 100
    # incorrect initialization
    loss_ema = 0
    loss_ema_alpha = 0.1
    # tied to `is_date`
    out_dir = jobs_dir / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    first_epoch = 0 if resume is None else resume['epoch']
    writing = first_epoch < epochs
    if writing:
        print(f"writing to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        if history is not None:
            shutil.copy(history, out_dir / loss_filename)
        ckpt(first_epoch)
    val = validation_loss()
    print("validation_loss:", "{:.3e}".format(val))
    if writing and not (out_dir / loss_filename).is_file():
        with open(out_dir / loss_filename, "a+") as fp:
            fp.write(json.dumps({"train": [], "val": val}) + "\n")
    for epoch in range(first_epoch, epochs):
        losses = []
        print(f"epoch {epoch} / {epochs}")
        pbar = tqdm.tqdm(dataset)
        for step, x in enumerate(pbar):
            logits, loss = model(
                    x["input_ids"][:, :-1].to(device_type),
                    x["input_ids"][:, 1:].to(device_type),
                    x["attention_mask"][:, 1:].to(device_type))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            loss_acc += loss_item
            loss_ema = loss_ema_alpha * loss_item + \
                    (1 - loss_ema_alpha) * loss_ema
            pbar.set_description("loss - {:.3e}".format(loss_ema))
            if (step + 1) % loss_acc_steps == 0:
                losses.append(loss_acc / loss_acc_steps)
                loss_acc = 0
        ckpt(epoch + 1)
        val = validation_loss()
        with open(out_dir / loss_filename, "a") as fp:
            fp.write(json.dumps({"train": losses, "val": val}) + "\n")
        print("validation_loss:", "{:.3e}".format(val))

latest_flag = object()
latest_job = lambda: max(filter(is_date, jobs_dir.iterdir()), key=basename)
is_digit = lambda x: ord('0') <= ord(x) <= ord('9')
basename = lambda x: x.name

def is_date(x):
    x = x.name.split("_")
    return len(x) == 6 and \
            all(len(i) == j for i, j in zip(x, (4, 2, 2, 2, 2, 2))) and \
            all(is_digit(j) for i in x for j in i)

def is_ckpt(x):
    x = x.name
    return x.startswith("ckpt-") and x.endswith(".pt") and len(x) > 8 and \
            all(is_digit(i) for i in x[5:-3])

def main(args):
    if args.ckpt is not None:
        if args.ckpt is latest_flag:
            args.ckpt = latest_job()
            args.ckpt = max(filter(is_ckpt, args.ckpt.iterdir()), key=basename)
            print(f"resuming from {args.ckpt}")
        ckpt = torch.load(args.ckpt)
        losses = pathlib.Path(args.ckpt).parents[0] / loss_filename
        losses = losses if losses.is_file() else None
        run(ckpt["config"], epochs=args.epochs, resume=ckpt, history=losses)
    else:
        config = next(iter(configs.values())) if args.preset is None else \
                configs[args.preset]
        run(config, epochs=args.epochs)

def plotter(args):
    import matplotlib.pyplot as plt
    if args.losses is latest_flag:
        args.losses = [latest_job() / loss_filename]
    for f in args.losses:
        with open(f, "r") as fp:
            history = list(map(json.loads, fp))
        history = {k: [i[k] for i in history] for k in history[0].keys()}
        history["steps"] = torch.asarray(list(map(len, history["train"])))
        history["steps"] = torch.cumsum(history["steps"], 0).tolist()
        history["train"] = sum(history["train"], [])
        if args.ema != 0:
            assert 0 < args.ema < 1
            acc = torch.asarray(0.01)
            cutoff = torch.maximum(torch.asarray(10), 1 + torch.ceil(
                    torch.log(acc) / torch.log(torch.asarray(args.ema))).int())
            weights = args.ema ** torch.flip(torch.arange(cutoff), (0,))
            smooth = F.conv1d(
                    torch.asarray(history["train"])[None, None, :],
                    weights[None, None, :],
                    padding=cutoff.item() - 1)[0, 0, :-cutoff + 1]
            norm = torch.cumsum(torch.flip(weights, (0,)), 0)
            smooth[:cutoff] /= norm
            smooth[cutoff:] /= norm[-1]
            history["train"] = smooth
        plt.semilogy(history["train"])
        plt.semilogy(history["steps"], history["val"])
    plt.show()

if __name__ == "__main__":
    from config import configs
    import argparse

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--preset", default=None)
    group.add_argument("--ckpt", default=None, nargs='?', const=latest_flag)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.set_defaults(caller=main)

    subparsers = parser.add_subparsers()

    plot = subparsers.add_parser("plot")
    plot.add_argument("losses", nargs="*", default=latest_flag)
    plot.add_argument("--ema", default=0., type=float)
    plot.set_defaults(caller=plotter)

    args = parser.parse_args()
    args.caller(args)
