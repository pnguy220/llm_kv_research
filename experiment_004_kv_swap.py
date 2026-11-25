"""Experiment 004: test whether operator-token K/V entries can be substituted.

The experiment uses addition and multiplication only. A cache is prefetched
through the target operator, the operator K/V is optionally replaced, and the
rest of the target prompt is recomputed before answer scoring and generation.
This is a functional cache intervention, not a test of prefill savings.
"""

import copy
import csv
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
OPERATIONS = ("add", "multiply")
SYMBOLS = {"add": "+", "multiply": "×"}
MAX_NEW_TOKENS = 24
SEED = 4004


def make_prompt(operation, a, b):
    return f"Calculate {a} {SYMBOLS[operation]} {b}. Answer with only the number:"


def expected_answer(operation, a, b):
    return a + b if operation == "add" else a * b


def find_operator_index(tokenizer, prompt, operation):
    encoded = tokenizer(prompt, return_tensors="pt", return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    char_index = prompt.index(SYMBOLS[operation])
    indices = [i for i, (start, end) in enumerate(offsets) if start <= char_index < end]
    if len(indices) != 1:
        raise ValueError(f"Operator must be one token: {prompt}")
    return encoded, indices[0]


def parse_answer(text, expected):
    """Parse a generated answer without mistaking echoed operands for it."""
    normalized = text.replace(",", "")
    # Prefer an explicit result after '=' or an answer label.
    matches = re.findall(r"(?:=|answer\s*(?:is|:)?)\s*\\?(-?\d+)", normalized, re.I)
    if matches:
        return int(matches[-1])
    # Accept a bare numeric answer only if generation begins with that number.
    match = re.match(r"\s*\\?(-?\d+)(?:\s|\.|$)", normalized)
    if match:
        return int(match.group(1))
    # If the model emits prose without a clear answer marker, do not guess.
    return None


def generate(model, tokenizer, logits, cache, device):
    generated = []
    token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    for index in range(MAX_NEW_TOKENS):
        token_id = token.item()
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id or index + 1 == MAX_NEW_TOKENS:
            break
        with torch.inference_mode():
            output = model(input_ids=token.to(device), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return tokenizer.decode(generated, skip_special_tokens=True)


def expected_answer_logprob(model, tokenizer, logits, cache, answer, device):
    """Teacher-force the expected numeric answer and sum its token log-prob."""
    answer_ids = tokenizer(" " + str(answer), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    score = 0.0
    for token_id in answer_ids[0]:
        score += F.log_softmax(logits[:, -1, :], dim=-1)[0, token_id].item()
        with torch.inference_mode():
            output = model(
                input_ids=token_id.view(1, 1),
                past_key_values=cache,
                use_cache=True,
            )
        cache = output.past_key_values
        logits = output.logits
    return score


def unique_path(directory, stem, extension):
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        path = directory / f"{stem}{suffix}.{extension}"
        if not path.exists():
            return path
        index += 1


def target_pairs():
    pairs = [(a, b) for a in range(2, 10) for b in range(2, 10) if a != b]
    random.Random(SEED).shuffle(pairs)
    return pairs[:12]


def prefill(model, input_ids, device):
    with torch.inference_mode():
        return model(input_ids=input_ids.to(device), use_cache=True)


def evaluate_continuation(model, tokenizer, suffix_output, expected, device):
    score = expected_answer_logprob(
        model,
        tokenizer,
        suffix_output.logits,
        copy.deepcopy(suffix_output.past_key_values),
        expected,
        device,
    )
    generated = generate(
        model,
        tokenizer,
        suffix_output.logits,
        copy.deepcopy(suffix_output.past_key_values),
        device,
    )
    parsed = parse_answer(generated, expected)
    return score, generated, parsed


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this experiment does not fall back to CPU.")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, local_files_only=True
    ).to(device)
    model.eval()
    num_layers = model.config.num_hidden_layers

    layer_sets = [(f"layer_{i}", (i,)) for i in range(num_layers)]
    for start in range(0, num_layers, 6):
        stop = min(start + 6, num_layers)
        layer_sets.append((f"layers_{start}_{stop - 1}", tuple(range(start, stop))))
    layer_sets.append(("all_layers", tuple(range(num_layers))))

    rows = []
    for a, b in target_pairs():
        for target_op in OPERATIONS:
            target_prompt = make_prompt(target_op, a, b)
            encoded, operator_index = find_operator_index(tokenizer, target_prompt, target_op)
            target_ids = encoded["input_ids"]
            expected = expected_answer(target_op, a, b)

            # Baseline full prefill and standard greedy continuation.
            base_output = prefill(model, target_ids, device)
            base_score = expected_answer_logprob(
                model, tokenizer, base_output.logits,
                copy.deepcopy(base_output.past_key_values), expected, device,
            )
            base_text = generate(
                model, tokenizer, base_output.logits,
                copy.deepcopy(base_output.past_key_values), device,
            )
            base_parsed = parse_answer(base_text, expected)
            rows.append({
                "target_operation": target_op,
                "source_operation": "none",
                "condition": "unmodified_baseline",
                "layer_set": "none",
                "layers_patched": "",
                "target_a": a,
                "target_b": b,
                "source_a": "",
                "source_b": "",
                "operator_token_position": operator_index,
                "expected_answer": expected,
                "expected_answer_logprob": base_score,
                "generated_text": base_text,
                "parsed_answer": base_parsed,
                "correct": base_parsed == expected,
                "device": device,
                "dtype": str(next(model.parameters()).dtype),
            })

            # Source and target share the exact prefix through the operator.
            # The same-operation source uses a different later operand; by
            # causality its operator K/V should match the target's own entry.
            source_b = 3 if b != 3 else 4
            source_prompts = {
                "same_operation_transfer": (target_op, source_b),
                "different_operation": ("multiply" if target_op == "add" else "add", source_b),
            }
            target_through_operator = prefill(
                model,
                target_ids[:, :operator_index + 1],
                device,
            ).past_key_values
            target_layer_cache = target_through_operator

            for condition, (source_op, source_b) in source_prompts.items():
                source_prompt = make_prompt(source_op, a, source_b)
                source_encoded, source_operator_index = find_operator_index(tokenizer, source_prompt, source_op)
                if source_operator_index != operator_index:
                    raise ValueError("Source and target operator positions do not align.")
                source_cache = prefill(
                    model,
                    source_encoded["input_ids"][:, :source_operator_index + 1],
                    device,
                ).past_key_values

                for layer_set_name, layers in layer_sets:
                    patched = copy.deepcopy(target_layer_cache)
                    for layer_index in layers:
                        patched.layers[layer_index].keys[:, :, operator_index, :] = source_cache.layers[
                            layer_index
                        ].keys[:, :, source_operator_index, :]
                        patched.layers[layer_index].values[:, :, operator_index, :] = source_cache.layers[
                            layer_index
                        ].values[:, :, source_operator_index, :]

                    # Self-copy is an intervention control: it should reproduce
                    # the target's own operator K/V at the selected layer(s).
                    if condition == "same_operation_transfer":
                        # source prefix is identical through the operator, so
                        # this is cross-prompt provenance with identical entry.
                        pass
                    suffix_ids = target_ids[:, operator_index + 1:]
                    with torch.inference_mode():
                        suffix_output = model(
                            input_ids=suffix_ids.to(device),
                            past_key_values=patched,
                            use_cache=True,
                        )
                    score, generated, parsed = evaluate_continuation(
                        model, tokenizer, suffix_output, expected, device
                    )
                    rows.append({
                        "target_operation": target_op,
                        "source_operation": source_op,
                        "condition": condition,
                        "layer_set": layer_set_name,
                        "layers_patched": ";".join(map(str, layers)),
                        "target_a": a,
                        "target_b": b,
                        "source_a": a,
                        "source_b": source_b,
                        "operator_token_position": operator_index,
                        "expected_answer": expected,
                        "expected_answer_logprob": score,
                        "generated_text": generated,
                        "parsed_answer": parsed,
                        "correct": parsed == expected,
                        "device": device,
                        "dtype": str(next(model.parameters()).dtype),
                    })

            # Explicit self-copy control, using the target's own cached K/V.
            for layer_set_name, layers in layer_sets:
                patched = copy.deepcopy(target_layer_cache)
                for layer_index in layers:
                    patched.layers[layer_index].keys[:, :, operator_index, :] = target_layer_cache.layers[
                        layer_index
                    ].keys[:, :, operator_index, :]
                    patched.layers[layer_index].values[:, :, operator_index, :] = target_layer_cache.layers[
                        layer_index
                    ].values[:, :, operator_index, :]
                with torch.inference_mode():
                    suffix_output = model(
                        input_ids=target_ids[:, operator_index + 1:].to(device),
                        past_key_values=patched,
                        use_cache=True,
                    )
                score, generated, parsed = evaluate_continuation(
                    model, tokenizer, suffix_output, expected, device
                )
                rows.append({
                    "target_operation": target_op,
                    "source_operation": target_op,
                    "condition": "self_copy_control",
                    "layer_set": layer_set_name,
                    "layers_patched": ";".join(map(str, layers)),
                    "target_a": a,
                    "target_b": b,
                    "source_a": a,
                    "source_b": b,
                    "operator_token_position": operator_index,
                    "expected_answer": expected,
                    "expected_answer_logprob": score,
                    "generated_text": generated,
                    "parsed_answer": parsed,
                    "correct": parsed == expected,
                    "device": device,
                    "dtype": str(next(model.parameters()).dtype),
                })

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = unique_path(results_dir, "exp04_operator_kv_swap", "csv")
    plot_path = unique_path(results_dir, "exp04_operator_kv_swap_accuracy", "png")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Compare answer exact-match rate by layer set. Baseline is a reference
    # line; it is repeated in the figure only as a horizontal aggregate.
    individual = [r for r in rows if r["layer_set"].startswith("layer_") and r["condition"] != "unmodified_baseline"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for condition in ("self_copy_control", "same_operation_transfer", "different_operation"):
        selected = [r for r in individual if r["condition"] == condition]
        grouped = {}
        for row in selected:
            grouped.setdefault(int(row["layers_patched"]), []).append(row["correct"])
        xs = sorted(grouped)
        ys = [sum(value is True for value in grouped[x]) / len(grouped[x]) for x in xs]
        ax.plot(xs, ys, marker=".", label=condition.replace("_", " "))
    baselines = [r["correct"] for r in rows if r["condition"] == "unmodified_baseline"]
    if baselines:
        ax.axhline(sum(value is True for value in baselines) / len(baselines), color="black", linestyle="--", label="baseline exact-match rate")
    ax.set(xlabel="Patched transformer layer", ylabel="Generated-answer exact-match rate", ylim=(0, 1.02), title="Operator K/V substitution by layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"target prompts: {len(OPERATIONS) * len(target_pairs())}; rows: {len(rows)}")
    print("Conditions: baseline, self-copy, same-op cross-suffix, different-op")
    print(f"saved results: {csv_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
