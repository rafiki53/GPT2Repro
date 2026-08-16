import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from rich import print as rprint
from hellaswag import render_example, iterate_examples

def rotate_half_and_partially_negate(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).                                      view(1, 1, config.block_size, config.block_size))
       
        head_dim = config.n_embd // config.n_head
        freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))  # head_dim // 2
        freq = torch.outer(torch.arange(config.block_size, dtype = torch.float32), freq)  # (block_size, head_dim // 2)
        freq = torch.cat((freq, freq), dim=-1).view(1, 1, config.block_size, head_dim)  # (1, 1, block_size, head_dim)
        self.register_buffer("cos", freq.cos(), persistent=False)
        self.register_buffer("sin", freq.sin(), persistent=False)

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim = 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        cos = self.cos[:, :, :T, :]
        sin = self.sin[:, :, :T, :]
        q = (q * cos + rotate_half_and_partially_negate(q) * sin)
        k = (k * cos + rotate_half_and_partially_negate(k) * sin)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # y = F.scaled_dot_product_attention(q, k, v, is_causal=True)


        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
        block_size: int = 128
        vocab_size: int = 50257  # 50,000 BPE merges + 256 + 1 EOT
        n_layer: int = 12
        n_head: int = 6
        n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std = std) 
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.size()
        assert T <= self.config.block_size, f"{T=}, {self.config.block_size=}"

        x = self.transformer.wte(idx)  # (B, T, n_embd)

        for block in self.transformer.h:
            x = block(x)  # (B, T, n_embd)

        x = self.transformer.ln_f(x)  # (B, T, n_embd)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        rprint(f"{len(decay_params)=}, {num_decay_params=}")
        rprint(f"{len(nodecay_params)=}, {num_nodecay_params=}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer


import tiktoken
import numpy as np

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T

        assert split in {'train', 'val'}

        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"{len(shards)=}, {split=}"
        rprint(f"{len(shards)=}, {split=}")

        self.load_shard(0)

    def load_shard(self, shard):
        self.current_shard = shard
        npt = np.load(self.shards[self.current_shard])
        self.tokens = torch.tensor(npt, dtype=torch.long)
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.load_shard(self.current_shard + 1)
        return x, y

def get_most_likely_row(tokens, mask, logits):
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens,  reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = (mask[..., 1:]).contiguous()
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    pred_norm = avg_loss.argmin()
    return pred_norm



torch.manual_seed(1337)
model = GPT(GPTConfig(vocab_size=50257))
model = torch.compile(model)
enc = tiktoken.get_encoding("gpt2")

total_batch_size = 16384 # 524288
B = 2
T = GPTConfig().block_size
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)
rprint(f"{GPTConfig()=}")
rprint(f"{total_batch_size=}, {B*T=}, {grad_accum_steps=}")


max_lr = 1.2e-3
min_lr = max_lr * 0.1
warmup_steps = 50
max_steps = 50000
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr)
train_loader = DataLoaderLite(B = B, T = GPTConfig().block_size, split="train")
val_loader = DataLoaderLite(B = B, T = GPTConfig().block_size, split="val")

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.text")
with open(log_file, "w") as f:
    pass

grad_norm_lr_accum = 0.0
for step in range(max_steps):
    if step % 250 == 0:
        t0 = time.time()
        model.eval()
        val_loader.load_shard(0)
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                __, loss = model(x, y)
                loss = loss/val_loss_steps
                val_loss_accum += loss.detach()
        t1 = time.time()
        rprint(f"{step=}, {val_loss_accum/math.log(2)=:.4f}, {grad_norm_lr_accum/math.log(2)=:.4f}")
        grad_norm_lr_accum = 0.0

        t0 = time.time()
        max_length = GPTConfig().block_size
        tokens = enc.encode("Coriolis force is")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0)
        sample_rng = torch.Generator()
        sample_rng.manual_seed(42)
        while tokens.size(1) < max_length:
            with torch.no_grad():
                logits, loss = model(tokens)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size)
                probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (B, 50)
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
                next_tokens = torch.gather(topk_indices, -1, ix)  # (B, 1)
                tokens = torch.cat((tokens, next_tokens), dim = 1)
        tokens = tokens[0, :max_length].tolist()
        decoded = enc.decode(tokens)
        t1 = time.time()
        print(decoded)

        t0 = time.time()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val", max_len = GPTConfig().block_size)):
            _, tokens, mask, label = render_example(example)
            with torch.no_grad():
                logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        acc_norm = num_correct_norm/num_total
        rprint(f"HellaSwag: {num_total=}, {acc_norm=}")
        with open(log_file, "a") as f:
            f.write(f"{step=} hella {acc_norm=}")
        rprint(f"Eval: {time.time()-t0=}")
        if step % 5000 == 0:
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                    'model': model.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss/log2': val_loss_accum.item()/math.log(2),
            }
            torch.save(checkpoint, checkpoint_path)

    t0 = time.time()
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for _ in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        logits, loss = model(x, y)
        loss = loss/grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    norm  = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    grad_norm_lr_accum = grad_norm_lr_accum + min(norm, 1.0) ** 2 * lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    tokens = train_loader.B * train_loader.T * grad_accum_steps
    t1 = time.time()
    rprint(f"{step=}, {loss_accum/math.log(2)=:6.2f}, {lr=:.4e}, {norm=:.4f}, {step*tokens=}, {t1-t0=}, {tokens/(t1 - t0)=:6.2f}")

import sys; sys.exit(0)

model.eval()

num_return_sequences = 5
max_length = 30
