#!/usr/bin/env python3
import argparse
import json
import math
import os
import pathlib
from dataclasses import dataclass

import numpy as np


HIDDEN_SIZE = 2880
HEAD_DIM = 64
ATTN_HEADS = 64
KV_HEADS = 8
Q_MULT = ATTN_HEADS // KV_HEADS
ATTN_VALUES = ATTN_HEADS * HEAD_DIM
SLIDING_WINDOW = 128
INITIAL_CONTEXT_LENGTH = 4096.0
ROPE_THETA = 150000.0
ROPE_SCALING_FACTOR = 32.0
ROPE_NTK_ALPHA = 1.0
ROPE_NTK_BETA = 32.0
EXPERTS = 32
INTERMEDIATE_SIZE = 2880
GATE_UP_VALUES = INTERMEDIATE_SIZE * 2
MXFP4_GROUPS = HIDDEN_SIZE // 32
MXFP4_BYTES_PER_GROUP = 16
SWIGLU_LIMIT = 7.0
SWIGLU_ALPHA = 1.702
FP4_VALUES = np.array(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=np.float32,
)


@dataclass
class TensorInfo:
    path: pathlib.Path
    data_start: int
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]


class SafeTensorStore:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.tensors: dict[str, TensorInfo] = {}
        for path in sorted(root.glob("model-*.safetensors")):
            self._index_file(path)
        if not self.tensors:
            raise RuntimeError(f"no model-*.safetensors files found in {root}")

    def _index_file(self, path: pathlib.Path) -> None:
        with path.open("rb") as handle:
            header_len = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(header_len))
        data_start = 8 + header_len
        for name, value in header.items():
            if name == "__metadata__":
                continue
            self.tensors[name] = TensorInfo(
                path=path,
                data_start=data_start,
                dtype=value["dtype"],
                shape=tuple(value["shape"]),
                offsets=tuple(value["data_offsets"]),
            )

    def tensor(self, name: str) -> np.ndarray:
        info = self._info(name)
        with info.path.open("rb") as handle:
            handle.seek(info.data_start + info.offsets[0])
            data = handle.read(info.offsets[1] - info.offsets[0])
        return self._decode(info, data).reshape(info.shape)

    def bf16_row(self, name: str, row: int) -> np.ndarray:
        info = self._info(name)
        if info.dtype != "BF16" or len(info.shape) != 2:
            raise RuntimeError(f"{name} is {info.dtype} rank {len(info.shape)}, expected BF16 rank 2")
        rows, cols = info.shape
        if row >= rows:
            raise RuntimeError(f"row {row} is outside {name} rows {rows}")
        row_bytes = cols * 2
        with info.path.open("rb") as handle:
            handle.seek(info.data_start + info.offsets[0] + row * row_bytes)
            data = handle.read(row_bytes)
        return bf16_to_f32(np.frombuffer(data, dtype="<u2"))

    def u8_slice(self, name: str, offset: int, length: int) -> np.ndarray:
        info = self._info(name)
        if info.dtype != "U8":
            raise RuntimeError(f"{name} is {info.dtype}, expected U8")
        with info.path.open("rb") as handle:
            handle.seek(info.data_start + info.offsets[0] + offset)
            data = handle.read(length)
        return np.frombuffer(data, dtype=np.uint8).copy()

    def _info(self, name: str) -> TensorInfo:
        try:
            return self.tensors[name]
        except KeyError as error:
            raise RuntimeError(f"tensor {name} not found") from error

    def _decode(self, info: TensorInfo, data: bytes) -> np.ndarray:
        if info.dtype == "BF16":
            return bf16_to_f32(np.frombuffer(data, dtype="<u2"))
        if info.dtype == "U8":
            return np.frombuffer(data, dtype=np.uint8).copy()
        raise RuntimeError(f"unsupported dtype {info.dtype}")


def bf16_to_f32(values: np.ndarray) -> np.ndarray:
    bits = values.astype(np.uint32) << 16
    return bits.view(np.float32).copy()


def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    scale = 1.0 / math.sqrt(float(np.mean(x.astype(np.float64) ** 2)) + 1e-5)
    return (x * np.float32(scale) * weight).astype(np.float32)


def matvec(store: SafeTensorStore, name: str, x: np.ndarray) -> np.ndarray:
    weight = store.tensor(name)
    return (weight @ x).astype(np.float32)


def add_bias(store: SafeTensorStore, name: str, x: np.ndarray) -> np.ndarray:
    return (x + store.tensor(name)).astype(np.float32)


def yarn_concentration_and_inv_freq() -> tuple[float, np.ndarray]:
    concentration = 0.1 * math.log(ROPE_SCALING_FACTOR) + 1.0
    d_half = HEAD_DIM / 2.0
    low = (
        d_half
        * math.log(INITIAL_CONTEXT_LENGTH / (ROPE_NTK_BETA * 2.0 * math.pi))
        / math.log(ROPE_THETA)
    )
    high = (
        d_half
        * math.log(INITIAL_CONTEXT_LENGTH / (ROPE_NTK_ALPHA * 2.0 * math.pi))
        / math.log(ROPE_THETA)
    )
    dims = np.arange(HEAD_DIM // 2, dtype=np.float32)
    freq = np.float32(ROPE_THETA) ** ((dims * 2.0) / np.float32(HEAD_DIM))
    interpolation = 1.0 / (np.float32(ROPE_SCALING_FACTOR) * freq)
    extrapolation = 1.0 / freq
    ramp = (dims - np.float32(low)) / np.float32(high - low)
    mask = 1.0 - np.clip(ramp, 0.0, 1.0)
    inv_freq = interpolation * (1.0 - mask) + extrapolation * mask
    return concentration, inv_freq.astype(np.float32)


def apply_rope(rows: np.ndarray, heads: int) -> np.ndarray:
    out = rows.copy()
    concentration, inv_freq = yarn_concentration_and_inv_freq()
    seq_len = out.shape[0]
    out = out.reshape(seq_len, heads, HEAD_DIM)
    for position in range(seq_len):
        theta = np.float32(position) * inv_freq
        cos = np.cos(theta).astype(np.float32) * np.float32(concentration)
        sin = np.sin(theta).astype(np.float32) * np.float32(concentration)
        first = out[position, :, : HEAD_DIM // 2].copy()
        second = out[position, :, HEAD_DIM // 2 :].copy()
        out[position, :, : HEAD_DIM // 2] = first * cos - second * sin
        out[position, :, HEAD_DIM // 2 :] = second * cos + first * sin
    return out.reshape(seq_len, heads * HEAD_DIM).astype(np.float32)


def attention(layer: int, q: np.ndarray, k: np.ndarray, v: np.ndarray, sinks: np.ndarray) -> np.ndarray:
    seq_len = q.shape[0]
    q = q.reshape(seq_len, KV_HEADS, Q_MULT, HEAD_DIM)
    k = k.reshape(seq_len, KV_HEADS, HEAD_DIM)
    v = v.reshape(seq_len, KV_HEADS, HEAD_DIM)
    window = SLIDING_WINDOW if layer % 2 == 0 else 0
    sm_scale = np.float32(1.0 / math.sqrt(HEAD_DIM))
    out = np.zeros((seq_len, KV_HEADS, Q_MULT, HEAD_DIM), dtype=np.float32)

    for pos in range(seq_len):
        key_start = max(0, pos + 1 - window) if window else 0
        for kv_head in range(KV_HEADS):
            for q_index in range(Q_MULT):
                head = kv_head * Q_MULT + q_index
                scores = np.array(
                    [
                        np.dot(q[pos, kv_head, q_index], k[key_pos, kv_head]) * sm_scale
                        for key_pos in range(key_start, pos + 1)
                    ],
                    dtype=np.float32,
                )
                sink = np.float32(sinks[head])
                max_score = np.maximum(np.max(scores), sink)
                exp_scores = np.exp(scores - max_score).astype(np.float32)
                denom = np.sum(exp_scores, dtype=np.float32) + np.exp(sink - max_score).astype(np.float32)
                weights = exp_scores / denom
                for offset, key_pos in enumerate(range(key_start, pos + 1)):
                    out[pos, kv_head, q_index] += weights[offset] * v[key_pos, kv_head]

    return out.reshape(seq_len, ATTN_VALUES)


def top_k_softmax(values: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argsort(-values, kind="stable")[:k]
    logits = values[indices]
    logits = logits - np.max(logits)
    weights = np.exp(logits).astype(np.float32)
    weights = weights / np.sum(weights, dtype=np.float32)
    return indices.astype(np.int64), weights.astype(np.float32)


def mxfp4_expert_matvec(
    store: SafeTensorStore,
    blocks_name: str,
    scales_name: str,
    expert: int,
    rows: int,
    x: np.ndarray,
) -> np.ndarray:
    blocks_per_expert = rows * MXFP4_GROUPS * MXFP4_BYTES_PER_GROUP
    scales_per_expert = rows * MXFP4_GROUPS
    blocks = store.u8_slice(blocks_name, expert * blocks_per_expert, blocks_per_expert)
    scales = store.u8_slice(scales_name, expert * scales_per_expert, scales_per_expert)
    blocks = blocks.reshape(rows, MXFP4_GROUPS, MXFP4_BYTES_PER_GROUP)
    scales = scales.reshape(rows, MXFP4_GROUPS).astype(np.int32) - 127

    low = blocks & np.uint8(0x0F)
    high = blocks >> np.uint8(4)
    decoded = np.empty((rows, MXFP4_GROUPS, 32), dtype=np.float32)
    decoded[:, :, 0::2] = FP4_VALUES[low]
    decoded[:, :, 1::2] = FP4_VALUES[high]
    decoded = np.ldexp(decoded, scales[:, :, None]).astype(np.float32)
    return np.einsum("rgi,gi->r", decoded, x.reshape(MXFP4_GROUPS, 32), dtype=np.float32)


def swiglu(values: np.ndarray) -> np.ndarray:
    x_glu = np.minimum(values[0::2], np.float32(SWIGLU_LIMIT))
    x_linear = np.clip(values[1::2], -SWIGLU_LIMIT, SWIGLU_LIMIT)
    out_glu = x_glu / (1.0 + np.exp(-np.float32(SWIGLU_ALPHA) * x_glu))
    return (out_glu * (x_linear + 1.0)).astype(np.float32)


def layer_moe(store: SafeTensorStore, layer: int, x: np.ndarray, experts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    prefix = f"model.layers.{layer}.mlp.experts"
    out = np.zeros(HIDDEN_SIZE, dtype=np.float32)
    for expert, weight in zip(experts, weights):
        gate_up = mxfp4_expert_matvec(
            store,
            f"{prefix}.gate_up_proj_blocks",
            f"{prefix}.gate_up_proj_scales",
            int(expert),
            GATE_UP_VALUES,
            x,
        )
        gate_up += store.bf16_row(f"{prefix}.gate_up_proj_bias", int(expert))
        activated = swiglu(gate_up)
        down = mxfp4_expert_matvec(
            store,
            f"{prefix}.down_proj_blocks",
            f"{prefix}.down_proj_scales",
            int(expert),
            HIDDEN_SIZE,
            activated,
        )
        down += store.bf16_row(f"{prefix}.down_proj_bias", int(expert))
        out += down * weight
    return out.astype(np.float32)


def sequence_layer(store: SafeTensorStore, layer: int, x: np.ndarray) -> np.ndarray:
    prefix = f"model.layers.{layer}"
    norm_weight = store.tensor(f"{prefix}.input_layernorm.weight")
    attn_input = np.stack([rms_norm(row, norm_weight) for row in x])

    q = np.stack(
        [
            add_bias(store, f"{prefix}.self_attn.q_proj.bias", matvec(store, f"{prefix}.self_attn.q_proj.weight", row))
            for row in attn_input
        ]
    )
    k = np.stack(
        [
            add_bias(store, f"{prefix}.self_attn.k_proj.bias", matvec(store, f"{prefix}.self_attn.k_proj.weight", row))
            for row in attn_input
        ]
    )
    v = np.stack(
        [
            add_bias(store, f"{prefix}.self_attn.v_proj.bias", matvec(store, f"{prefix}.self_attn.v_proj.weight", row))
            for row in attn_input
        ]
    )

    q = apply_rope(q, ATTN_HEADS)
    k = apply_rope(k, KV_HEADS)
    attn = attention(layer, q, k, v, store.tensor(f"{prefix}.self_attn.sinks"))
    projected = np.stack(
        [
            add_bias(store, f"{prefix}.self_attn.o_proj.bias", matvec(store, f"{prefix}.self_attn.o_proj.weight", row))
            for row in attn
        ]
    )
    residual = (x + projected).astype(np.float32)

    post_norm_weight = store.tensor(f"{prefix}.post_attention_layernorm.weight")
    out = []
    for row in residual:
        router_input = rms_norm(row, post_norm_weight)
        router = add_bias(store, f"{prefix}.mlp.router.bias", matvec(store, f"{prefix}.mlp.router.weight", router_input))
        experts, weights = top_k_softmax(router, 4)
        moe = layer_moe(store, layer, router_input, experts, weights)
        out.append((row + moe).astype(np.float32))
    return np.stack(out)


def selected_logits(store: SafeTensorStore, hidden: np.ndarray, tokens: list[int]) -> list[tuple[int, float]]:
    out = []
    for token in tokens:
        row = store.bf16_row("lm_head.weight", token)
        out.append((token, float(np.dot(row, hidden))))
    return out


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.expanduser("~/.please/weights"))
    parser.add_argument("--tokens", default="200006,1428,200008,64614")
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--logit-tokens", default="277,8526,387,263,278,289,10581,1808")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    store = SafeTensorStore(pathlib.Path(args.weights))
    tokens = parse_csv_ints(args.tokens)
    logit_tokens = parse_csv_ints(args.logit_tokens)

    x = np.stack([store.bf16_row("model.embed_tokens.weight", token) for token in tokens])
    embedding_final_first8 = x[-1, :8].tolist()
    layer_checkpoints = []
    for layer in range(args.layers):
        x = sequence_layer(store, layer, x)
        final = x[-1]
        layer_checkpoints.append(
            {
                "layer": layer,
                "final_l2": float(np.linalg.norm(final)),
                "final_mean": float(np.mean(final)),
                "final_first8": final[:8].tolist(),
            }
        )

    final = rms_norm(x[-1], store.tensor("model.norm.weight"))
    selected = selected_logits(store, final, logit_tokens)
    selected = [{"token": token, "logit": logit} for token, logit in selected]

    if args.json:
        print(
            json.dumps(
                {
                    "weights": str(pathlib.Path(args.weights).expanduser()),
                    "tokens": tokens,
                    "layers": args.layers,
                    "embedding_final_first8": embedding_final_first8,
                    "layer_checkpoints": layer_checkpoints,
                    "final_norm_first8": final[:8].tolist(),
                    "selected_logits": selected,
                },
                indent=2,
            )
        )
        return

    print("python numpy oracle:")
    print(f"- weights: {args.weights}")
    print(f"- tokens: {tokens}")
    print(f"- layers: {args.layers}")
    print(f"- embedding final first 8: {embedding_final_first8}")
    for checkpoint in layer_checkpoints:
        print(
            f"- layer {checkpoint['layer']}: final_l2 {checkpoint['final_l2']:.7f}, "
            f"final_mean {checkpoint['final_mean']:.7f}, final_first8 {checkpoint['final_first8']}"
        )

    print(f"- final_norm first 8: {final[:8].tolist()}")
    print("- selected logits:")
    for logit in selected:
        print(f"  - token {logit['token']}: {logit['logit']:.7f}")


if __name__ == "__main__":
    main()
