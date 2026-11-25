"""Experiment 005: patch operator/final hidden activations across layers.

This mirrors the peer's activation-patching conditions while retaining this
project's Qwen2.5-0.5B base model on CUDA/FP16. It patches residual-stream
activations during a teacher-forced prompt+answer pass; it does not patch KV.
"""

import csv
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
TEMPLATE = "Calculate {a} {op} {b} = "
SYMBOLS = {"multiply": "×", "add": "+"}
OTHER = {"multiply": "add", "add": "multiply"}
ANSWER = {"multiply": lambda a, b: a * b, "add": lambda a, b: a + b}
OPS = ("multiply", "add")
POSITIONS = ("operator", "final")
PATCH_SETS = {"operator": ("operator",), "final": ("final",), "both": ("operator", "final")}
SOURCES = ("paired_other", "average_other", "average_same")
SEED = 5005
LOW, HIGH = 2, 20
CANONICAL_PAIRS = 30
TEST_PAIRS = 30
CHUNK_SIZE = 24


def make_prompt(a, operation, b):
    return TEMPLATE.format(a=a, op=SYMBOLS[operation], b=b)


def token_positions(tokenizer, prompt, operation):
    encoded = tokenizer(prompt, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    char_pos = prompt.index(SYMBOLS[operation])
    matches = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
               if start <= char_pos < end]
    if len(matches) != 1:
        raise ValueError(f"Could not locate one operator token in {prompt!r}")
    return {"operator": matches[0], "final": len(ids) - 1}, ids


def answer_metrics(model, tokenizer, prompt, answer, patch_spec=None):
    """Teacher-force answer tokens; optionally patch activation at one layer."""
    prompt_ids = tokenizer(prompt)["input_ids"]
    answer_ids = tokenizer(str(answer), add_special_tokens=False)["input_ids"]
    ids = torch.tensor([prompt_ids + answer_ids], device="cuda")
    handle = None
    if patch_spec is not None:
        layer, patch_vectors = patch_spec

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            for token_position, vector in patch_vectors.items():
                hidden[:, token_position, :] = vector.to(hidden.device, hidden.dtype)
            return output

        target = model.model.embed_tokens if layer == 0 else model.model.layers[layer - 1]
        handle = target.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            hidden = model.model(ids).last_hidden_state[:, len(prompt_ids) - 1:]
            logits = model.lm_head(hidden).float()
    finally:
        if handle is not None:
            handle.remove()
    logprobs = F.log_softmax(logits, dim=-1)
    score = sum(logprobs[0, k, token].item() for k, token in enumerate(answer_ids))
    exact = all(logits[0, k].argmax().item() == token for k, token in enumerate(answer_ids))
    return exact, score


def hidden_at_positions(model, tokenizer, prompt, positions):
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    with torch.inference_mode():
        states = model(ids, output_hidden_states=True, use_cache=False).hidden_states
    return {
        pos: torch.stack([states[layer][0, index].detach() for layer in range(24)])
        for pos, index in positions.items()
    }


def unique_path(directory, stem, ext):
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        path = directory / f"{stem}{suffix}.{ext}"
        if not path.exists():
            return path
        index += 1


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this experiment does not fall back to CPU.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, local_files_only=True
    ).to("cuda")
    model.eval()

    pairs = [(a, b) for a in range(LOW, HIGH + 1) for b in range(a + 1, HIGH + 1)]
    random.Random(SEED).shuffle(pairs)
    canonical, tests = [], []
    # Keep only pairs for which the base model scores the exact answer as its
    # greedy continuation for both operations and both operand orders.
    for a, b in pairs:
        clean = True
        for x, y in ((a, b), (b, a)):
            for op in OPS:
                ok, _ = answer_metrics(model, tokenizer, make_prompt(x, op, y), ANSWER[op](x, y))
                clean &= ok
        if clean:
            (canonical if len(canonical) < CANONICAL_PAIRS else tests).append((a, b))
        if len(canonical) >= CANONICAL_PAIRS and len(tests) >= TEST_PAIRS:
            break
    if len(canonical) < CANONICAL_PAIRS or len(tests) < TEST_PAIRS:
        raise RuntimeError(
            f"Only found {len(canonical)} canonical and {len(tests)} clean test pairs; "
            "reduce requested counts or expand the operand range."
        )

    # Cache clean canonical hidden states for average donors, and build the
    # per-test paired-other donor from the same numbers with the other op.
    canonical_states = {op: {pos: [] for pos in POSITIONS} for op in OPS}
    for a, b in canonical:
        for x, y in ((a, b), (b, a)):
            for op in OPS:
                prompt = make_prompt(x, op, y)
                positions, _ = token_positions(tokenizer, prompt, op)
                states = hidden_at_positions(model, tokenizer, prompt, positions)
                for pos in POSITIONS:
                    canonical_states[op][pos].append(states[pos])
    averages = {
        op: {pos: torch.stack(canonical_states[op][pos]).mean(dim=0) for pos in POSITIONS}
        for op in OPS
    }

    plan = [(source, patch_set, layer)
            for source in SOURCES
            for patch_set in PATCH_SETS
            for layer in range(24)]
    rows = []
    started = time.time()
    for a, b in tests:
        for x, y in ((a, b), (b, a)):
            for op in OPS:
                prompt = make_prompt(x, op, y)
                positions, _ = token_positions(tokenizer, prompt, op)
                answer = ANSWER[op](x, y)
                other_answer = ANSWER[OTHER[op]](x, y)
                clean_ok, clean_lp = answer_metrics(model, tokenizer, prompt, answer)
                other_clean_ok, other_clean_lp = answer_metrics(model, tokenizer, prompt, other_answer)

                paired_prompt = make_prompt(x, OTHER[op], y)
                paired_pos, _ = token_positions(tokenizer, paired_prompt, OTHER[op])
                paired = hidden_at_positions(model, tokenizer, paired_prompt, paired_pos)
                vectors = {
                    "paired_other": paired,
                    "average_other": averages[OTHER[op]],
                    "average_same": averages[op],
                }
                for source, patch_set, layer in plan:
                    selected = {
                        positions[pos]: vectors[source][pos][layer]
                        for pos in PATCH_SETS[patch_set]
                    }
                    correct, answer_lp = answer_metrics(
                        model, tokenizer, prompt, answer,
                        patch_spec=(layer, selected),
                    )
                    alt_correct, alt_lp = answer_metrics(
                        model, tokenizer, prompt, other_answer,
                        patch_spec=(layer, selected),
                    )
                    rows.append({
                        "model": MODEL_NAME,
                        "device": "cuda",
                        "dtype": str(next(model.parameters()).dtype),
                        "a": x,
                        "b": y,
                        "operation": op,
                        "answer": answer,
                        "other_operation_answer": other_answer,
                        "source": source,
                        "patch_set": patch_set,
                        "layer": layer,
                        "operator_position": positions["operator"],
                        "final_position": positions["final"],
                        "clean_answer_correct": clean_ok,
                        "clean_answer_logprob": clean_lp,
                        "clean_other_answer_correct": other_clean_ok,
                        "clean_other_answer_logprob": other_clean_lp,
                        "patched_answer_correct": correct,
                        "patched_answer_logprob": answer_lp,
                        "patched_other_answer_correct": alt_correct,
                        "patched_other_answer_logprob": alt_lp,
                    })
        print(f"completed pair {a},{b}; elapsed {time.time() - started:.0f}s", flush=True)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = unique_path(results_dir, "exp05_activation_patch", "csv")
    plot_path = unique_path(results_dir, "exp05_activation_patch_accuracy", "png")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
    for row, op in enumerate(OPS):
        for col, patch_set in enumerate(("operator", "final", "both")):
            ax = axes[row, col]
            for source in SOURCES:
                selected = [r for r in rows if r["operation"] == op and r["patch_set"] == patch_set and r["source"] == source]
                acc = [sum(r["patched_answer_correct"] for r in selected if r["layer"] == layer) /
                       sum(r["layer"] == layer for r in selected) for layer in range(24)]
                ax.plot(range(24), acc, label=source.replace("_", " "))
            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, label="clean target = 1.0")
            ax.set_title(f"{op}: patch {patch_set}")
            ax.set_xlabel("Layer (activation entering this layer)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("Original answer exact-match rate")
    axes[1, 0].set_ylabel("Original answer exact-match rate")
    axes[0, 0].legend(fontsize="small")
    fig.suptitle("Experiment 005: hidden-state activation patching")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    print(f"clean canonical pairs: {len(canonical)}; clean held-out pairs: {len(tests)}")
    print(f"saved results: {csv_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
