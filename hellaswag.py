import os
import json
import requests
import tiktoken
from tqdm import tqdm
import math
import jax
import jax.numpy as jnp
import numpy as np
import optax


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream = True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):
        print(f"Downloading: {data_url=}, {data_filename=}")
        download_file(data_url, data_filename)

def render_example(example):
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    data = {
        "label": label,
        "ctx_tokens": enc.encode(ctx),
        "ending_tokens": [],
    }

    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)  # prepending " " because of the GPT2 tokenizer
        tok_rows.append(data["ctx_tokens"] + end_tokens)
        mask_rows.append([0]*len(data["ctx_tokens"]) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    max_len = max(len(row) for row in tok_rows)
    tokens = np.zeros((4, max_len), dtype=np.int32)
    mask = np.zeros((4, max_len), dtype=np.int32)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = tok_row
        mask[i, :len(mask_row)] = mask_row

    return data, jnp.array(tokens), jnp.array(mask), label

def iterate_examples(split, max_len = math.inf):
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            if len(enc.encode(example["ctx"])) + max([len(enc.encode(i)) for i in example["endings"]]) < max_len:
                yield example

def get_most_likely_row(
    tokens: jax.Array, mask: jax.Array, logits: jax.Array
) -> int:
    shift_logits = logits[:, :-1, :]
    shift_tokens = tokens[:, 1:]
    shift_mask = mask[:, 1:]
    shift_losses = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits, shift_tokens)
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = jnp.sum(masked_shift_losses, axis=-1)
    avg_loss = sum_loss / jnp.maximum(1.0, jnp.sum(shift_mask, axis=-1))
    pred_norm = int(jnp.argmin(avg_loss))
    return pred_norm
