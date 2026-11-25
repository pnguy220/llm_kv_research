"""Layerwise linear probes for arithmetic operation information.

The model is run only on CUDA in FP16. Probe training/testing also stay on
CUDA. Operand pairs are split before prompts are made, so test pairs never
appear in training. The probed vector is always the final shared Answer: token.
"""

import csv
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ANSWER_SUFFIX = "\nAnswer:"
NUMBERS = tuple(range(2, 22))
OPERATIONS = ("add", "subtract", "multiply", "divide")
SYMBOLS = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷"}
NATURAL_NAMES = {
    "add": "sum",
    "subtract": "difference",
    "multiply": "product",
    "divide": "quotient",
}
SEED = 1337
PROBE_EPOCHS = 100
LEARNING_RATE = 0.05


def make_operand_split():
    """Create deterministic, disjoint train/test sets of ordered pairs."""
    pairs = [(a, b) for a in NUMBERS for b in NUMBERS if a != b]
    random.Random(SEED).shuffle(pairs)
    test_pairs = pairs[::5]
    test_set = set(test_pairs)
    train_pairs = [pair for pair in pairs if pair not in test_set]
    return train_pairs, test_pairs


def make_prompt(operation, a, b, style):
    if style == "symbolic":
        body = f"Calculate {a} {SYMBOLS[operation]} {b}."
    elif style == "natural":
        body = f"What is the {NATURAL_NAMES[operation]} of {a} and {b}?"
    else:
        raise ValueError(f"Unknown prompt style: {style}")
    return body + ANSWER_SUFFIX


def unique_path(directory, stem, extension):
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        path = directory / f"{stem}{suffix}.{extension}"
        if not path.exists():
            return path
        index += 1


def fit_linear_probe(train_x, train_y, num_classes):
    """Fit a simple multinomial linear classifier on the GPU."""
    probe = torch.nn.Linear(train_x.shape[1], num_classes, device="cuda")
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE)
    for _ in range(PROBE_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    return probe


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this script does not fall back to CPU.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_pairs, test_pairs = make_operand_split()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda")
    model.eval()

    # Build two evaluation conditions. The cross-wording condition trains on
    # symbolic prompts and tests natural-language prompts with unseen operands.
    examples = []
    for split_name, pairs, style in (
        ("train", train_pairs, "symbolic"),
        ("test_same_style", test_pairs, "symbolic"),
        ("test_cross_wording", test_pairs, "natural"),
    ):
        for a, b in pairs:
            for label, operation in enumerate(OPERATIONS):
                examples.append({
                    "split": split_name,
                    "operand_a": a,
                    "operand_b": b,
                    "operation": operation,
                    "label": label,
                    "style": style,
                    "text": make_prompt(operation, a, b, style),
                })

    vectors = []
    with torch.inference_mode():
        for example in examples:
            inputs = tokenizer(example["text"], return_tensors="pt").to("cuda")
            output = model(**inputs, output_hidden_states=True, use_cache=False)
            # Every prompt ends at the same Answer: token. This makes the
            # comparison position consistent while retaining prompt context.
            vectors.append(torch.stack([
                state[0, -1, :].float() for state in output.hidden_states
            ]))
            del output
    vectors = torch.stack(vectors)  # [examples, hidden-state index, hidden size]

    train_indices = [i for i, ex in enumerate(examples) if ex["split"] == "train"]
    same_indices = [i for i, ex in enumerate(examples) if ex["split"] == "test_same_style"]
    cross_indices = [i for i, ex in enumerate(examples) if ex["split"] == "test_cross_wording"]
    labels = torch.tensor([ex["label"] for ex in examples], device="cuda")

    result_rows = []
    num_layers = vectors.shape[1]
    for layer_index in range(num_layers):
        train_x = vectors[train_indices, layer_index]
        # Normalize using training data only. This avoids scale differences
        # dominating optimization and does not leak test statistics.
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
        train_x = (train_x - mean) / std
        train_y = labels[train_indices]
        probe = fit_linear_probe(train_x, train_y, len(OPERATIONS))

        for condition, indices in (
            ("heldout_operands_same_wording", same_indices),
            ("heldout_operands_cross_wording", cross_indices),
        ):
            test_x = (vectors[indices, layer_index] - mean) / std
            test_y = labels[indices]
            with torch.no_grad():
                predictions = probe(test_x).argmax(dim=1)
            accuracy = (predictions == test_y).float().mean().item()
            result_rows.append({
                "model": MODEL_NAME,
                "device": "cuda",
                "dtype": str(next(model.parameters()).dtype),
                "probe": "linear_softmax",
                "condition": condition,
                "layer_index": layer_index,
                "layer_name": "embedding_output" if layer_index == 0 else f"transformer_layer_{layer_index}",
                "token_position": "final Answer: token",
                "accuracy": accuracy,
                "chance_accuracy": 1 / len(OPERATIONS),
                "train_examples": len(train_indices),
                "test_examples": len(indices),
                "train_operand_pairs": len(train_pairs),
                "test_operand_pairs": len(test_pairs),
                "epochs": PROBE_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
            })

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = unique_path(output_dir, "exp03_operation_probe", "csv")
    examples_path = unique_path(output_dir, "exp03_probe_examples", "csv")
    plot_path = unique_path(output_dir, "exp03_operation_probe_accuracy", "png")

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    with examples_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "operand_a", "operand_b", "operation", "label", "style", "text"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(examples)

    fig, ax = plt.subplots(figsize=(10, 5))
    for condition in ("heldout_operands_same_wording", "heldout_operands_cross_wording"):
        selected = [r for r in result_rows if r["condition"] == condition]
        ax.plot([r["layer_index"] for r in selected], [r["accuracy"] for r in selected], marker=".", label=condition.replace("_", " "))
    ax.axhline(0.25, color="black", linestyle="--", linewidth=1, label="chance (25%)")
    ax.set(xlabel="Hidden-state index (0 = embedding output)", ylabel="Operation classification accuracy", ylim=(0, 1.02), title="Operation probe across layers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"train operand pairs: {len(train_pairs)}; held-out pairs: {len(test_pairs)}")
    print(f"probe token position: final shared Answer: token; hidden-state indices: {num_layers}")
    print(f"saved results: {results_path}")
    print(f"saved examples: {examples_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
