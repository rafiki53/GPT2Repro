import dataclasses
import io
import math
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
import time

from absl import app
from absl import flags
from etils import epath
from flax import nnx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from rich import print as rprint
import tiktoken

import hellaswag

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "data_root",
    os.path.join(os.path.expanduser("~"), "edu_fineweb10B"),
    "Directory containing the tokenized FineWebEdu shards.")
flags.DEFINE_string("checkpoint_dir", "", "Optional checkpoint path.")

@dataclasses.dataclass
class Config:

    # maximum sequence length
    t: int = 1024  # VL 1024
    # embedding dimension
    c: int = 768
    # number of attention heads
    n: int = 6
    # number of residual stream multiplexing dimensions
    m: int = 4
    # in-memory batch size, i.e. the number of sequences actually processed in parallel
    # on the device
    b: int = 4
    # vocabulary size
    v: int = 50257
    # number of transformer layers
    num_layers: int = 12
    # number of sequences processed per optimizer step
    step_batch_size: int = 512  # VL 512
    # save a checkpoint if step is divisible by this number
    save_interval_steps: int = 512
    # total number of tokens to train for
    train_tokens: int = 2**24
    # if <q, k> = 0 for all other keys and <q, k*> = this value, then the
    # attention softmax for k* will be 0.5. The attn_scale multiplied onto
    # attention scores is calculated from this value.
    attn_score_for_half_softmax: float = 0.45
    
    # head dimesnsion equal to embedding dimension divided by the number of heads
    h: int= dataclasses.field(init=False)# = 128  # c // n
    # number of 2D ROPE subspaces equal to half the head dimension
    r: int= dataclasses.field(init=False)# = 64  # h // 2
    # bound for uniform matrix initialization
    init_bound: float= dataclasses.field(init=False)# = 0.03125  # 3.0**0.5 * 0.5 * (c**-0.5)
    # number of gradient accumulation steps (in-memory batches) per optimizer step
    grad_accumulation_steps: int= dataclasses.field(init=False)# = 512
    # number of tokens processed per optimizer step
    step_tokens: int= dataclasses.field(init=False)# = 2**19  # step_batch_size * t
    # total number of optimizer steps
    train_steps: int= dataclasses.field(init=False)# = 2**5  # train_tokens // step_tokens
    # multiplied onto attention scores before the softmax
    # In "Attention is All You Need" this was 1 / h
    attn_scale: float= dataclasses.field(init=False)# = 0.12033805218 # ln(t) / (h * attn_score_for_half_softmax)


    def __post_init__(self):
        self.init_bound = 3.0**0.5 * 0.5 * (self.c**-0.5)
        assert self.c % self.n == 0
        self.h = self.c // self.n
        assert self.step_batch_size % self.b == 0
        self.grad_accumulation_steps = self.step_batch_size // self.b
        
        self.step_tokens = self.step_batch_size * self.t
        assert self.h % 2 == 0
        self.r = self.h // 2
        self.train_steps = self.train_tokens // self.step_tokens
        self.attn_scale=math.log(self.t) / (self.h * self.attn_score_for_half_softmax)
        
cfg = Config()
rprint(cfg)

def uni_init(rngs: nnx.Rngs, shape: tuple[int, ...]) -> jax.Array:
    return jax.random.uniform(
            rngs.params(), shape, minval=-cfg.init_bound, maxval=cfg.init_bound
    )

def rms_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt(
            jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps
    )

def rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)

class MuonParam(nnx.Param):

    def __jax_array__(self):
        return self[...]

    max_lr: float = 0.03
    wd: float = 0.01

class EmbeddingParam(nnx.Param):
    max_lr: float = 0.3
    wd: float = 0.01

class OutputHeadParam(nnx.Param):
    max_lr: float = 0.004
    wd: float = 0.01

class MuxParam(nnx.Param):
    max_lr: float = 0.004
    wd: float = 0.0

class MuxUp(nnx.Module):

    def __init__(
            self,
            rngs: nnx.Rngs,
    ):
        self.w = MuxParam(uni_init(rngs, (cfg.m,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x[..., None] * jax.nn.softmax(self.w)

class MuxDown(nnx.Module):

    def __init__(
            self,
            rngs: nnx.Rngs,
    ):
        self.w = MuxParam(uni_init(rngs, (cfg.m,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.sum(x * jax.nn.softmax(self.w), axis = -1)

class MuonLinear(nnx.Module):

    def __init__(self, matrix: jax.Array):
        self.w = MuonParam(matrix)

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.matmul(x, self.w)

class CausalSelfAttention(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        self.q_nch = MuonParam(uni_init(rngs, (cfg.n, cfg.c, cfg.h)))
        self.k_nch = MuonParam(uni_init(rngs, (cfg.n, cfg.c, cfg.h)))
        self.v_cc = MuonParam(uni_init(rngs, (cfg.c, cfg.c)))
        self.out_linear_cc = MuonLinear(jnp.zeros((cfg.c, cfg.c)))

        # Precompute sin and cos for rotation matrix multiplication
        active_frequencies_34r = 1.0 / (
            (cfg.t / (math.pi / 8))
            ** (jnp.arange(0, cfg.r * 3 // 4, dtype=jnp.float32) / cfg.r)
        )
        channel_frequencies_r = jnp.concatenate([
            active_frequencies_34r,
            jnp.zeros((cfg.r // 4), dtype=jnp.float32),
        ])
        angles_tr = jnp.outer(
            jnp.arange(cfg.t, dtype=jnp.float32), channel_frequencies_r
        )
        self.cos_1t1r = jnp.cos(angles_tr)[None, :, None, :]
        self.sin_1t1r = jnp.sin(angles_tr)[None, :, None, :]

    def __call__(self, x_btc: jax.Array, doc_mask_b1tt: jax.Array) -> jax.Array:
        b, t, _ = x_btc.shape

        q_btnh = jnp.einsum("btc, nch -> btnh", x_btc, self.q_nch)
        k_btnh = jnp.einsum("btc, nch -> btnh", x_btc, self.k_nch)
        v_btnh = jnp.matmul(x_btc, self.v_cc).reshape(b, t, cfg.n, cfg.h)

        q_btnh = rms_norm(q_btnh)
        k_btnh = rms_norm(k_btnh)

        q_btnh = rope(q_btnh, cos=self.cos_1t1r[:, :t, :, :],
                      sin = self.sin_1t1r[:, :t, :, :])
        k_btnh = rope(k_btnh, cos=self.cos_1t1r[:, :t, :, :],
                      sin = self.sin_1t1r[:, :t, :, :])

        q_bnth = jnp.swapaxes(q_btnh, 1, 2)
        k_bnth = jnp.swapaxes(k_btnh, 1, 2)
        v_bnth = jnp.swapaxes(v_btnh, 1, 2)

        scores_bntt = (
            jnp.matmul(q_bnth, jnp.swapaxes(k_bnth, -1, -2)) * cfg.attn_scale)
        mask_11tt = jnp.tril(jnp.ones((t, t), dtype=jnp.bool_))[None, None, :, :]
        attention_bntt = jnn.softmax(
            jnp.where(mask_11tt & doc_mask_b1tt, scores_bntt, -jnp.inf), axis = -1)
        heads_out_bnth = jnp.matmul(attention_bntt, v_bnth)
        heads_out_btnh = jnp.swapaxes(heads_out_bnth, 1, 2)
        heads_out_btc = heads_out_btnh.reshape(b, t, cfg.c)

        return self.out_linear_cc(heads_out_btc)

class TransformerBlock(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        self.mux_in_m1 = MuxDown(rngs)
        self.attention = CausalSelfAttention(rngs)
        self.mux_attention_out_1m = MuxUp(rngs)
        self.mux_linear_in_m1 = MuxDown(rngs)
        self.linear_up_c4c = MuonLinear(uni_init(rngs, (cfg.c, 4 * cfg.c)))
        self.linear_down_4cc = MuonLinear(jnp.zeros((4 * cfg.c, cfg.c)))
        self.mux_out_1m = MuxUp(rngs)

    def __call__(self, x_btcm: jax.Array, doc_mask_b1tt: jax.Array) -> jax.Array:
        x_btc = self.mux_in_m1(x_btcm)
        x_btcm = x_btcm + self.mux_attention_out_1m(
                self.attention(rms_norm(x_btc), doc_mask_b1tt)
        )
        x_btc = self.mux_linear_in_m1(x_btcm)
        return x_btcm + self.mux_out_1m(
            self.linear_down_4cc(
                jnp.square(jnn.relu(self.linear_up_c4c(rms_norm(x_btc))))
            )
        )

class GPT(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        self.embedding_vc = EmbeddingParam(
            jax.random.normal(rngs.params(), (cfg.v, cfg.c))
        )
        self.blocks = nnx.List(
            [TransformerBlock(rngs) for _ in range(cfg.num_layers)]
        )
        self.output_mux_m1 = MuxDown(rngs)
        self.output_head_cv = OutputHeadParam(jnp.zeros((cfg.c, cfg.v)))
    
    def __call__(self, idx_bt: jax.Array) -> jax.Array:
        doc_ids_bt = jnp.cumsum(idx_bt == 50256, axis=-1)
        doc_mask_b1tt = doc_ids_bt[:, None, :, None] == doc_ids_bt[:, None, None, :]
        x_btc = rms_norm(self.embedding_vc[...][idx_bt])
        x_btcm = jnp.broadcast_to(x_btc[..., None], (*x_btc.shape, cfg.m))
        for block in self.blocks:
            x_btcm = block(x_btcm, doc_mask_b1tt)
        x_btc = self.output_mux_m1(x_btcm)
        y_btv = jnp.matmul(rms_norm(x_btc), self.output_head_cv)
        return y_btv

def newton_shulz5(g: jax.Array) -> jax.Array:
    x = g / (jnp.linalg.norm(g, axis=(-2, -1), keepdims=True) + 1e-7)

    is_tall = x.shape[-2] > x.shape[-1]
    if is_tall:
        x = jnp.swapaxes(x, -1, -2)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        x_xt = jnp.matmul(x, jnp.swapaxes(x, -1, -2))
        x = a * x + jnp.matmul(b * x_xt + c * jnp.matmul(x_xt, x_xt), x)

    if is_tall:
        x = jnp.swapaxes(x, -1, -2)
    return x

class Muon(nnx.Module):

    def __init__(self, model: nnx.Module):
        self.momentum = nnx.Variable(
            jax.tree.map(jnp.zeros_like, nnx.state(model, MuonParam))
        )

    def get_update(self, params: nnx.State, grads: nnx.State, step: int) -> nnx.State:
        b1 = jnp.minimum(0.95, 0.85 + 0.00025 * step)
        self.momentum.set_value(
            jax.tree.map(
                lambda m, g: b1 * m + (1.0 - b1) * g,
                self.momentum.get_value(),
                grads,
            )
        )

        def update_p(p, g, m):
            nesterov_update = b1 * m + (1.0 - b1) * g
            update = newton_shulz5(nesterov_update)
            cautious_wd_mask = (update * p) > 0
            lr = p.max_lr * jnp.minimum(1.0, 2.0 - 2.0 * step / cfg.train_steps)
            p = p * (1.0 - lr * p.wd * cautious_wd_mask)
            return (p
                    - lr
                    * jnp.maximum(1.0, update.shape[-1] / update.shape[-2]) ** 0.5
                    * update
            )

        return jax.tree.map(
                update_p,
                params,
                grads,
                self.momentum.get_value(),
                is_leaf=lambda x: isinstance(x, nnx.Variable))

class AdamW(nnx.Module):

    def __init__(self, model: nnx.Module):
        self.b1 = 0.8
        self.b2 = 0.95
        adam_params = nnx.state(model, nnx.All(nnx.Param, nnx.Not(MuonParam)))
        self.momentum = nnx.Variable(jax.tree.map(jnp.zeros_like, adam_params))
        self.rmsprop = nnx.Variable(jax.tree.map(jnp.zeros_like, adam_params))

    def get_update(
        self, params: nnx.State, grads: nnx.State, step: int) -> nnx.State:
        bias_c1 = 1.0 - (self.b1**step)
        bias_c2 = 1.0 - (self.b2**step)

        self.momentum.set_value(
            jax.tree.map(
                lambda m, g: self.b1 * m + (1.0 - self.b1) * g,
                self.momentum.get_value(),
                grads,
            )
        )
        self.rmsprop.set_value(
            jax.tree.map(
                lambda v, g: self.b2 * v + (1.0 - self.b2) * (g**2),
                self.rmsprop.get_value(),
                grads,
            )
        )

        def update_p(p, m, v):
            update = (m / bias_c1) / (jnp.sqrt(v / bias_c2) + 1e-10)
            cautious_wd_mask = (update * p) > 0
            lr = p.max_lr * jnp.minimum(1.0, 2.0 - 2.0 * step / cfg.train_steps)
            p = p * (1.0 - lr * p.wd * cautious_wd_mask)
            return p - lr * update

        return jax.tree.map(
            update_p,
            params,
            self.momentum.get_value(),
            self.rmsprop.get_value(),
            is_leaf=lambda x: isinstance(x, nnx.Variable),
        )

class AdamWAndMuon(nnx.Module):

    def __init__(self, model: nnx.Module):
        self.muon = Muon(model)
        self.adamw = AdamW(model)
    
    @nnx.jit
    def update(self, model: nnx.Module, grads: nnx.State, step: int):
        nnx.update(
            model,
            self.muon.get_update(
                nnx.state(model, MuonParam), grads.filter(MuonParam), step
            ),
        )
        nnx.update(
            model,
            self.adamw.get_update(
                nnx.state(model, nnx.All(nnx.Param, nnx.Not(MuonParam))),
                grads.filter(nnx.All(nnx.Param, nnx.Not(MuonParam))),
                step,
            ),
        )


class DataLoader:
    def __init__(self, split: str, data_root: str):
        assert split in {'train', 'val'}
        self.split = split
        self.data_root = epath.Path(data_root)

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root {data_root} not found.")

        self.shards = sorted(
                p
                for p in self.data_root.iterdir()
                if split in p.name and p.name.endswith(".npy")
        )
        rprint(f"{len(self.shards)=}, {split=}")
        self.load_shard(0)

    def load_shard(self, shard: int):
        self.current_shard = shard
        npt = np.load(io.BytesIO(self.shards[shard].read_bytes()))
        self.tokens = npt.astype(np.int32)
        self.current_pos = 0

    def next_batch(self):
        buf = self.tokens[self.current_pos : self.current_pos+cfg.b*cfg.t + 1]
        x = (buf[:-1]).reshape(cfg.b, cfg.t)
        y = (buf[1:]).reshape(cfg.b, cfg.t)
        self.current_pos += cfg.b * cfg.t
        if self.current_pos + (cfg.b*cfg.t+1) > len(self.tokens):
            if self.shards:
                self.load_shard((self.current_shard + 1) % len(self.shards))
            else:
                self.current_pos = 0
        return jnp.array(x), jnp.array(y)

@nnx.jit
def run_in_memory_batch(mdl: GPT, token_ids: jax.Array, targets: jax.Array):
    def loss_fn(model):
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(
                model(token_ids), targets
            )
        )
    return nnx.value_and_grad(loss_fn)(mdl)
    
@nnx.jit
def run_in_memory_and_combine_steps(mdl: GPT, token_ids: jax.Array, targets: jax.Array, accum_grads: GPT,train_loss:float):
    loss, grads = run_in_memory_batch(mdl, token_ids, targets)
    if accum_grads is None:
        accum_grads = jax.tree.map(
            lambda g: g / cfg.grad_accumulation_steps, grads
        )
    else:
        accum_grads = jax.tree.map(
            lambda a, g: a + (g /  cfg.grad_accumulation_steps),
            accum_grads,
            grads,
        )
    train_loss += loss / cfg.grad_accumulation_steps
    return accum_grads,train_loss
    
def main(argv: list[str]):
    del argv  # Unused

    rngs = nnx.Rngs(params=0, generate=1)
    enc = tiktoken.get_encoding("gpt2")
    model = GPT(rngs)
    optimizer = AdamWAndMuon(model)
    train_loader = DataLoader(split="train", data_root=FLAGS.data_root)
    val_loader = DataLoader(split="val", data_root=FLAGS.data_root)

    ckpt_mgr = None
    start_step = 1
    if FLAGS.checkpoint_dir:
        ckpt_mgr = ocp.CheckpointManager(
            epath.Path(FLAGS.checkpoint_dir),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=2,
                save_interval_steps=cfg.save_interval_steps,
                create=True,
            ),
        )
        latest = ckpt.mgr.latest_step()
        if latest is not None:
            restored = ckpt_mgr.restore(
                latest,
                args=ocp.args.Composite(
                    model.ocp.args.StandardRestore(nnx.state(model, nnx.Param)),
                    optimizer=ocp.args.StandardRestore(nnx.state(optimizer)),
                    loader=ocp.args.JsonRestore(),
                ),
            )
            nnx.update(model, restored["model"])
            nnx.update(optimizer, restored["optimizer"])
            train_loader.load_shard(restored["loader"]["shard"])
            train_loader.current_pos = restored["loader"]["pos"]
            start_step = latest + 1
            rprint(f"{start_step=}")
    rprint(f"Before any steps{jax.local_devices()[0].memory_stats()=}")
    next_x_train, next_y_train = None, None
    for step in range(start_step, cfg.train_steps + 1):
        t0 = time.time()
        accum_grads=  None
        train_loss = 0.0
        if step % cfg.save_interval_steps == 1:
            assert next_x_train is None and next_y_train is None
            next_x_train, next_y_train = train_loader.next_batch()
        for i in range(cfg.grad_accumulation_steps):
            #loss, grads = run_in_memory_batch(model, next_x_train, next_y_train)
            # Don't load the next batch just before saving a checkpoint
            accum_grads,train_loss = run_in_memory_and_combine_steps(model,next_x_train,next_y_train, accum_grads,train_loss) 
            if (
                step % cfg.save_interval_steps == 0
                and i == cfg.grad_accumulation_steps - 1
            ):
                next_x_train, next_y_train = None, None
            else:
                next_x_train, next_y_train = train_loader.next_batch()
                
            #jax.block_until_ready(grads)
            # if accum_grads is None:
                # accum_grads = jax.tree.map(
                    # lambda g: g / cfg.grad_accumulation_steps, grads
                # )
            # else:
                # accum_grads = jax.tree.map(
                    # lambda a, g: a + (g /  cfg.grad_accumulation_steps),
                    # accum_grads,
                    # grads,
                # )
            # train_loss += loss / cfg.grad_accumulation_steps
            
        if step % 1 == 0:
            rprint(
                f"{step=}, {train_loss/math.log(2)=:.4f} bits/token,"
                f" {step*cfg.step_tokens=},"
                f" {cfg.step_tokens/(time.time()-t0)=:.2f} tokens/s"
            )
        
        rprint(f"OPTIMIZER IS ABOUT TO RUN")
        rprint(f"{jax.local_devices()[0].memory_stats()=}")
        optimizer.update(model, accum_grads, jnp.asarray(step))

        if step % 1 == 0:
            t0 = time.time()
            if val_loader.shards:
                val_loader.load_shard(0)
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                loss, _ = run_in_memory_batch(model, x, y)
                val_loss_accum += float(loss) / val_loss_steps
            t1 = time.time()
            rprint(f"{step=}, {val_loss_accum/math.log(2)=:.4f} bits/token, "
                   f"{time.time()-t0=:.2f} s")

            max_gen_len = 24 
            prompt_tokens = enc.encode("Coriolis force is")
            tokens = jnp.array([prompt_tokens], dtype=jnp.int32)

            while tokens.shape[1] < max_gen_len:
                logits_1tv = model(tokens)
                probs_1v = jnn.softmax(logits_1tv[:, -1, :], axis=-1)
                topk_probs, topk_indices = jax.lax.top_k(probs_1v, k=50)  # 1, 50
                token_idx = jax.random.categorical(rngs.generate(), topk_probs)  # 1, 1
                next_token = jnp.take_along_axis(
                    topk_indices, token_idx[:, None], axis=1
                )
                tokens = jnp.concatenate([tokens, next_token], axis = 1)  # 1, t + 1
            
            decoded = enc.decode([int(t) for t in tokens[0, :max_gen_len]])
            rprint(f"{decoded=}")

#            try:
            num_correct_norm = 0
            num_total = 0
            for _, example in enumerate(
                    hellaswag.iterate_examples("val", max_len=max_gen_len)
            ):
                _, ex_tokens, mask, label = hellaswag.render_example(example)
                logits = model(ex_tokens)
                pred_norm = hellaswag.get_most_likely_row(ex_tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)

            if num_total > 0:
                acc_norm = num_correct_norm/num_total
                rprint(f"HellaSwag: {num_total=}, {acc_norm=:.4f}")
#            except Exception as hellaswag_exception:
#                rprint(f"{hellaswag_exception=}")

        if ckpt_mgr:
            ckpt_mgr.save(
                step,
                args=ocp.args.Composite(
                    model=ocp.args.StandardSave(nnx.state(model, nnx.Param)),
                    optimizer=ocp.args.StandardSave(nnx.state(optimizer)),
                    loader=ocp.args.JsonSave({
                        "shard": train_loader.current_shard,
                        "pos": train_loader.current_pos,
                    }),
                ),
            )

    if ckpt_mgr:
        ckpt_mgr.wait_until_finished()
        ckpt_mgr.close()

if __name__ == "__main__":
    app.run(main)
