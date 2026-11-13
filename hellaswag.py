import math
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tiktoken
from datasets import DownloadConfig, load_dataset

enc = tiktoken.get_encoding("gpt2")


def render_example(example):
  ctx = example["ctx"]
  # Fix 1: HF stores label as a string ("0".."3"), whereas raw JSONL used an int
  label = int(example["label"]) if example["label"] != "" else -1
  endings = example["endings"]

  data = {
      "label": label,
      "ctx_tokens": enc.encode(ctx),
      "ending_tokens": [],
  }

  tok_rows = []
  mask_rows = []
  for end in endings:
    end_tokens = enc.encode(
        " " + end
    )  # prepending " " because of the GPT2 tokenizer
    tok_rows.append(data["ctx_tokens"] + end_tokens)
    mask_rows.append([0] * len(data["ctx_tokens"]) + [1] * len(end_tokens))
    data["ending_tokens"].append(end_tokens)

  max_len = max(len(row) for row in tok_rows)
  tokens = np.zeros((4, max_len), dtype=np.int32)
  mask = np.zeros((4, max_len), dtype=np.int32)
  for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
    tokens[i, : len(tok_row)] = tok_row
    mask[i, : len(mask_row)] = mask_row

  return data, jnp.array(tokens), jnp.array(mask), label


@lru_cache(maxsize=1)
def get_dataset():
  try:
    return load_dataset(
        "Rowan/hellaswag",
        download_config=DownloadConfig(local_files_only=True),
    )
  except Exception:
    print("Downloading HellaSwag from Hugging Face...")
    return load_dataset("Rowan/hellaswag")


def iterate_examples(split, max_len=math.inf):
  # Your training script uses "val"; Hugging Face uses "validation".
  split = {"val": "validation"}.get(split, split)
  if split not in ("train", "validation", "test"):
    raise ValueError(f"Unknown HellaSwag split: {split}")

  dataset = get_dataset()[split]

  for example in dataset:
    # Fix 2: Filter by token sequence length (matching your model's block_size),
    # rather than cutting off after `max_len` rows.
    ctx_len = len(enc.encode(example["ctx"]))
    max_end_len = max(len(enc.encode(" " + end)) for end in example["endings"])
    if ctx_len + max_end_len <= max_len:
      yield example


def get_most_likely_row(
    tokens: jax.Array, mask: jax.Array, logits: jax.Array
) -> int:
  shift_logits = logits[:, :-1, :]
  shift_tokens = tokens[:, 1:]
  shift_mask = mask[:, 1:]
  shift_losses = optax.softmax_cross_entropy_with_integer_labels(
      shift_logits, shift_tokens
  )
  masked_shift_losses = shift_losses * shift_mask
  sum_loss = jnp.sum(masked_shift_losses, axis=-1)
  avg_loss = sum_loss / jnp.maximum(1.0, jnp.sum(shift_mask, axis=-1))
  pred_norm = int(jnp.argmin(avg_loss))
  return pred_norm