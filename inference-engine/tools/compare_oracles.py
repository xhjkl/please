#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ORACLE = ROOT / "inference-engine" / "tools" / "python_oracle.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tokens", default="200006,1428,200008,64614")
    parser.add_argument("--layers", default="24")
    parser.add_argument("--logit-tokens", default="277,8526,387,263,278,289,10581,1808")
    parser.add_argument("--abs-tol", type=float, default=1e-2)
    parser.add_argument("--rel-tol", type=float, default=1e-6)
    args = parser.parse_args()

    python = run_json(
        [
            args.python,
            str(PYTHON_ORACLE),
            "--tokens",
            args.tokens,
            "--layers",
            args.layers,
            "--logit-tokens",
            args.logit_tokens,
            "--json",
        ]
    )
    rust = run_json(
        [
            "cargo",
            "run",
            "-p",
            "inference-engine",
            "--bin",
            "cpu_oracle",
            "--quiet",
            "--",
            "--tokens",
            args.tokens,
            "--layers",
            args.layers,
            "--logit-tokens",
            args.logit_tokens,
            "--json",
        ]
    )

    mismatches = []
    compare_int("layers", rust, python, mismatches)
    compare_exact_list("tokens", rust, python, mismatches)
    compare_list("embedding_final_first8", rust, python, args.abs_tol, args.rel_tol, mismatches)
    compare_list("final_norm_first8", rust, python, args.abs_tol, args.rel_tol, mismatches)
    compare_logits(rust, python, args.abs_tol, args.rel_tol, mismatches)
    compare_layers(rust, python, args.abs_tol, args.rel_tol, mismatches)

    print("oracle comparison:")
    print(f"- layers: {rust['layers']}")
    print(f"- tokens: {rust['tokens']}")
    print(f"- selected logits: {len(rust['selected_logits'])}")
    print(f"- abs tolerance: {args.abs_tol}")
    print(f"- rel tolerance: {args.rel_tol}")
    if mismatches:
        print(f"- status: failed ({len(mismatches)} mismatches)")
        for mismatch in mismatches[:24]:
            print(f"  - {mismatch}")
        raise SystemExit(1)
    print("- status: ok")


def run_json(cmd: list[str]) -> dict:
    result = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed: {' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(result.stdout)


def compare_int(path: str, rust: dict, python: dict, mismatches: list[str]) -> None:
    if rust[path] != python[path]:
        mismatches.append(f"{path}: rust {rust[path]} != python {python[path]}")


def compare_exact_list(path: str, rust: dict, python: dict, mismatches: list[str]) -> None:
    if rust[path] != python[path]:
        mismatches.append(f"{path}: rust {rust[path]} != python {python[path]}")


def compare_list(
    path: str,
    rust: dict,
    python: dict,
    abs_tol: float,
    rel_tol: float,
    mismatches: list[str],
) -> None:
    left = rust[path]
    right = python[path]
    if len(left) != len(right):
        mismatches.append(f"{path}: rust len {len(left)} != python len {len(right)}")
        return
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if not close(left_value, right_value, abs_tol, rel_tol):
            mismatches.append(
                f"{path}[{index}]: rust {left_value:.9g}, python {right_value:.9g}, "
                f"abs_delta {abs(left_value - right_value):.9g}"
            )


def compare_logits(
    rust: dict,
    python: dict,
    abs_tol: float,
    rel_tol: float,
    mismatches: list[str],
) -> None:
    rust_logits = {entry["token"]: entry["logit"] for entry in rust["selected_logits"]}
    python_logits = {entry["token"]: entry["logit"] for entry in python["selected_logits"]}
    if rust_logits.keys() != python_logits.keys():
        mismatches.append(
            f"selected_logits tokens differ: rust {sorted(rust_logits)}, python {sorted(python_logits)}"
        )
        return
    for token in sorted(rust_logits):
        left = rust_logits[token]
        right = python_logits[token]
        if not close(left, right, abs_tol, rel_tol):
            mismatches.append(
                f"selected_logits[{token}]: rust {left:.9g}, python {right:.9g}, "
                f"abs_delta {abs(left - right):.9g}"
            )


def compare_layers(
    rust: dict,
    python: dict,
    abs_tol: float,
    rel_tol: float,
    mismatches: list[str],
) -> None:
    rust_layers = rust["layer_checkpoints"]
    python_layers = python["layer_checkpoints"]
    if len(rust_layers) != len(python_layers):
        mismatches.append(f"layer_checkpoints len: rust {len(rust_layers)} != python {len(python_layers)}")
        return
    for rust_layer, python_layer in zip(rust_layers, python_layers):
        layer = rust_layer["layer"]
        if layer != python_layer["layer"]:
            mismatches.append(f"layer index: rust {layer} != python {python_layer['layer']}")
            continue
        for key in ["final_l2", "final_mean"]:
            left = rust_layer[key]
            right = python_layer[key]
            if not close(left, right, abs_tol, rel_tol):
                mismatches.append(
                    f"layer {layer} {key}: rust {left:.9g}, python {right:.9g}, "
                    f"abs_delta {abs(left - right):.9g}"
                )
        for index, (left, right) in enumerate(zip(rust_layer["final_first8"], python_layer["final_first8"])):
            if not close(left, right, abs_tol, rel_tol):
                mismatches.append(
                    f"layer {layer} final_first8[{index}]: rust {left:.9g}, python {right:.9g}, "
                    f"abs_delta {abs(left - right):.9g}"
                )


def close(left: float, right: float, abs_tol: float, rel_tol: float) -> bool:
    delta = abs(left - right)
    scale = max(abs(left), abs(right), 1.0)
    return delta <= abs_tol or delta <= rel_tol * scale


if __name__ == "__main__":
    main()
