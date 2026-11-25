"""Test operation and operand decodability across inputs and wording.

The Qwen model runs only on CUDA/FP16. A linear probe is trained on symbolic
prompts and operand pairs disjoint from all test pairs. All probes read the
final shared Answer: token, making layerwise comparisons position-consistent.
"""

import csv
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ANSWER_SUFFIX = "\nAnswer:"
NUMBERS = tuple(range(2, 22))
OPERATIONS = ("add", "subtract", "multiply", "divide")
SYMBOLS = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷"}
WORDS = {
    "add": ("sum", "add", "add"),
    "subtract": ("difference", "subtract", "subtract"),
    "multiply": ("product", "multiply", "multiply"),
    "divide": ("quotient", "divide", "divide"),
}
NATURAL_STYLES = ("noun_question", "verb_instruction", "verb_question")
SEED = 2403
EPOCHS = 100
LEARNING_RATE = 0.05


def split_operand_pairs():
    pairs = [(a, b) for a in NUMBERS for b in NUMBERS if a != b]
    random.Random(SEED).shuffle(pairs)
    test_pairs = pairs[::5]
    test_set = set(test_pairs)
    train_pairs = [pair for pair in pairs if pair not in test_set]
    return train_pairs, test_pairs


def format_prompt(operation, a, b, style):
    noun, verb, _ = WORDS[operation]
    if style == "symbolic":
        body = f"Calculate {a} {SYMBOLS[operation]} {b}."
    elif style == "noun_question":
        body = f"What is the {noun} of {a} and {b}?"
    elif style == "verb_instruction":
        body = f"{verb.capitalize()} {a} by {b}." if operation in ("multiply", "divide") else f"{verb.capitalize()} {a} and {b}."
    elif style == "verb_question":
        body = f"What do you get when you {verb} {a} and {b}?"
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


def fit_probe(train_x, train_y, classes, seed):
    torch.manual_seed(seed)
    probe = torch.nn.Linear(train_x.shape[1], classes, device="cuda")
    optimizer = torch.optim.Adam(probe.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    return probe


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; this script does not fall back to CPU.")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    train_pairs, test_pairs = split_operand_pairs()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, local_files_only=True
    ).to("cuda")
    model.eval()

    examples = []
    for split, pairs, styles in (
        ("train", train_pairs, ("symbolic",)),
        ("test", test_pairs, ("symbolic",) + NATURAL_STYLES),
    ):
        for a, b in pairs:
            for operation_id, operation in enumerate(OPERATIONS):
                for style in styles:
                    examples.append({
                        "split": split,
                        "operand_a": a,
                        "operand_b": b,
                        "operand_a_label": a - NUMBERS[0],
                        "operand_b_label": b - NUMBERS[0],
                        "operation": operation,
                        "operation_label": operation_id,
                        "style": style,
                        "text": format_prompt(operation, a, b, style),
                    })

    # Save only the selected token vector from each layer, on CPU, to limit
    # GPU memory use. Probe fitting and scoring move one layer's data to CUDA.
    vectors = []
    with torch.inference_mode():
        for example in examples:
            tokens = tokenizer(example["text"], return_tensors="pt").to("cuda")
            output = model(**tokens, output_hidden_states=True, use_cache=False)
            vectors.append(torch.stack([
                state[0, -1, :].float().cpu() for state in output.hidden_states
            ]))
            del output
    vectors = torch.stack(vectors)

    train_indices = [i for i, ex in enumerate(examples) if ex["split"] == "train"]
    result_rows = []
    num_layers = vectors.shape[1]
    for layer in range(num_layers):
        train_x = vectors[train_indices, layer].to("cuda")
        mean = train_x.mean(dim=0, keepdim=True)
        scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
        train_x = (train_x - mean) / scale

        # Three tasks read the same activation: operation, first operand, and
        # second operand. Evaluation styles all use the same held-out pairs.
        task_specs = (
            ("operation", "operation_label", len(OPERATIONS)),
            ("operand_a", "operand_a_label", len(NUMBERS)),
            ("operand_b", "operand_b_label", len(NUMBERS)),
        )
        for task, label_field, class_count in task_specs:
            train_y = torch.tensor(
                [examples[i][label_field] for i in train_indices], device="cuda"
            )
            probe = fit_probe(train_x, train_y, class_count, SEED + layer)
            eval_styles = ("symbolic",) + NATURAL_STYLES
            for style in eval_styles:
                test_indices = [
                    i for i, ex in enumerate(examples)
                    if ex["split"] == "test" and ex["style"] == style
                ]
                test_x = (vectors[test_indices, layer].to("cuda") - mean) / scale
                test_y = torch.tensor(
                    [examples[i][label_field] for i in test_indices], device="cuda"
                )
                with torch.no_grad():
                    prediction = probe(test_x).argmax(dim=1)
                result_rows.append({
                    "model": MODEL_NAME,
                    "device": "cuda",
                    "dtype": str(next(model.parameters()).dtype),
                    "task": task,
                    "condition": "heldout_operands_" + style,
                    "layer_index": layer,
                    "layer_name": "embedding_output" if layer == 0 else f"transformer_layer_{layer}",
                    "token_position": "final Answer: token",
                    "accuracy": (prediction == test_y).float().mean().item(),
                    "chance_accuracy": 1 / class_count,
                    "class_count": class_count,
                    "train_examples": len(train_indices),
                    "test_examples": len(test_indices),
                    "train_operand_pairs": len(train_pairs),
                    "test_operand_pairs": len(test_pairs),
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "seed": SEED,
                })

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = unique_path(results_dir, "exp03_generalization", "csv")
    examples_path = unique_path(results_dir, "exp03_generalization_examples", "csv")
    plot_path = unique_path(results_dir, "exp03_generalization_accuracy", "png")
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    with examples_path.open("w", newline="", encoding="utf-8") as f:
        fields = list(examples[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(examples)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for axis, task in zip(axes, ("operation", "operand_a", "operand_b")):
        for style in ("symbolic",) + NATURAL_STYLES:
            selected = [r for r in result_rows if r["task"] == task and r["condition"] == "heldout_operands_" + style]
            axis.plot([r["layer_index"] for r in selected], [r["accuracy"] for r in selected], marker=".", label=style.replace("_", " "))
        chance = 0.25 if task == "operation" else 1 / len(NUMBERS)
        axis.axhline(chance, color="black", linestyle="--", linewidth=1, label=f"chance ({chance:.0%})")
        axis.set(title=task.replace("_", " "), xlabel="Hidden-state index", ylabel="Accuracy", ylim=(0, 1.02))
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    fig.suptitle("Layerwise decoding on held-out operand pairs")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"training pairs: {len(train_pairs)}; held-out pairs: {len(test_pairs)}")
    print("token position: final shared Answer: token")
    print(f"saved results: {results_path}")
    print(f"saved examples: {examples_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
