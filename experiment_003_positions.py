"""Experiment 003 extension: locate operation information by token position.

Only addition and multiplication are used. The Qwen model and linear probes
run on CUDA; there is no CPU fallback. Probes are trained on symbolic prompts
and evaluated on disjoint operand pairs in symbolic and natural wording.
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
OPERATIONS = ("add", "multiply")
LABELS = {operation: index for index, operation in enumerate(OPERATIONS)}
STYLES = ("symbolic", "noun_question", "verb_instruction")
POSITIONS = ("operation_cue", "operand_a", "operand_b", "answer_marker")
SEED = 3004
EPOCHS = 100
LEARNING_RATE = 0.05


def split_pairs():
    pairs = [(a, b) for a in NUMBERS for b in NUMBERS if a != b]
    random.Random(SEED).shuffle(pairs)
    test_pairs = pairs[::5]
    held_out = set(test_pairs)
    return [pair for pair in pairs if pair not in held_out], test_pairs


def make_prompt(operation, a, b, style):
    if style == "symbolic":
        body = f"Calculate {a} {'+' if operation == 'add' else '×'} {b}."
    elif style == "noun_question":
        noun = "sum" if operation == "add" else "product"
        body = f"What is the {noun} of {a} and {b}?"
    elif style == "verb_instruction":
        body = f"Add {a} and {b}." if operation == "add" else f"Multiply {a} by {b}."
    else:
        raise ValueError(f"Unknown prompt style: {style}")
    return body + ANSWER_SUFFIX


def find_span(text, phrase, start_at=0):
    start = text.index(phrase, start_at)
    return start, start + len(phrase)


def token_index_for_span(offsets, span):
    start, end = span
    indices = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
    if not indices:
        raise ValueError(f"No tokenizer token overlaps character span {span}.")
    # For multi-token operation phrases, use the final token in the phrase.
    return indices[-1]


def unique_path(directory, stem, extension):
    index = 1
    while True:
        suffix = "" if index == 1 else f"_{index}"
        path = directory / f"{stem}{suffix}.{extension}"
        if not path.exists():
            return path
        index += 1


def fit_probe(train_x, train_y, seed):
    torch.manual_seed(seed)
    probe = torch.nn.Linear(train_x.shape[1], len(OPERATIONS), device="cuda")
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
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    if not tokenizer.is_fast:
        raise SystemExit("A fast tokenizer with character offsets is required.")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, local_files_only=True
    ).to("cuda")
    model.eval()
    train_pairs, test_pairs = split_pairs()

    examples = []
    for split, pairs, styles in (
        ("train", train_pairs, ("symbolic",)),
        ("test", test_pairs, STYLES),
    ):
        for a, b in pairs:
            for operation in OPERATIONS:
                for style in styles:
                    text = make_prompt(operation, a, b, style)
                    body_end = text.index(ANSWER_SUFFIX)
                    if style == "symbolic":
                        cue = "+" if operation == "add" else "×"
                    elif style == "noun_question":
                        cue = "sum" if operation == "add" else "product"
                    else:
                        cue = "Add" if operation == "add" else "Multiply"
                    spans = {
                        "operation_cue": find_span(text, cue),
                        "operand_a": find_span(text, str(a)),
                        "operand_b": find_span(text, str(b), start_at=text.index(str(a)) + len(str(a))),
                        "answer_marker": find_span(text, ANSWER_SUFFIX.strip(), start_at=body_end),
                    }
                    encoding = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
                    offsets = [tuple(pair) for pair in encoding.pop("offset_mapping")[0].tolist()]
                    selected = {role: token_index_for_span(offsets, span) for role, span in spans.items()}
                    examples.append({
                        "split": split,
                        "operand_a": a,
                        "operand_b": b,
                        "operation": operation,
                        "label": LABELS[operation],
                        "style": style,
                        "text": text,
                        "operation_cue_token": tokenizer.convert_ids_to_tokens(encoding["input_ids"][0, selected["operation_cue"]].item()),
                        "operand_a_token": tokenizer.convert_ids_to_tokens(encoding["input_ids"][0, selected["operand_a"]].item()),
                        "operand_b_token": tokenizer.convert_ids_to_tokens(encoding["input_ids"][0, selected["operand_b"]].item()),
                        "selected_positions": selected,
                        "inputs": encoding,
                    })

    # Capture only the four role-specific vectors per layer on CPU; the probe
    # fits one layer at a time on CUDA to keep GPU memory use manageable.
    for example in examples:
        with torch.inference_mode():
            output = model(**example["inputs"].to("cuda"), output_hidden_states=True, use_cache=False)
        example["vectors"] = {
            role: torch.stack([
                state[0, example["selected_positions"][role], :].float().cpu()
                for state in output.hidden_states
            ])
            for role in POSITIONS
        }
        del output

    train_indices = [i for i, ex in enumerate(examples) if ex["split"] == "train"]
    result_rows = []
    num_layers = examples[0]["vectors"][POSITIONS[0]].shape[0]
    for role in POSITIONS:
        for layer in range(num_layers):
            train_x = torch.stack([examples[i]["vectors"][role][layer] for i in train_indices]).to("cuda")
            mean = train_x.mean(dim=0, keepdim=True)
            scale = train_x.std(dim=0, keepdim=True).clamp_min(1e-5)
            train_x = (train_x - mean) / scale
            train_y = torch.tensor([examples[i]["label"] for i in train_indices], device="cuda")
            probe = fit_probe(train_x, train_y, SEED + layer)

            for style in STYLES:
                test_indices = [i for i, ex in enumerate(examples) if ex["split"] == "test" and ex["style"] == style]
                test_x = torch.stack([examples[i]["vectors"][role][layer] for i in test_indices]).to("cuda")
                test_x = (test_x - mean) / scale
                test_y = torch.tensor([examples[i]["label"] for i in test_indices], device="cuda")
                with torch.no_grad():
                    predicted = probe(test_x).argmax(dim=1)
                result_rows.append({
                    "model": MODEL_NAME,
                    "device": "cuda",
                    "dtype": str(next(model.parameters()).dtype),
                    "task": "operation_classification_add_vs_multiply",
                    "token_position_role": role,
                    "condition": "heldout_operands_" + style,
                    "layer_index": layer,
                    "layer_name": "embedding_output" if layer == 0 else f"transformer_layer_{layer}",
                    "accuracy": (predicted == test_y).float().mean().item(),
                    "chance_accuracy": 0.5,
                    "train_operand_pairs": len(train_pairs),
                    "test_operand_pairs": len(test_pairs),
                    "train_examples": len(train_indices),
                    "test_examples": len(test_indices),
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "seed": SEED,
                })

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = unique_path(results_dir, "exp03_token_positions", "csv")
    examples_path = unique_path(results_dir, "exp03_token_positions_examples", "csv")
    plot_path = unique_path(results_dir, "exp03_token_positions_accuracy", "png")
    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=result_rows[0].keys())
        writer.writeheader()
        writer.writerows(result_rows)
    with examples_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["split", "operand_a", "operand_b", "operation", "label", "style", "text", "operation_cue_token", "operand_a_token", "operand_b_token", "selected_positions"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for example in examples:
            writer.writerow({key: example[key] for key in fields})

    fig, axes = plt.subplots(1, len(POSITIONS), figsize=(18, 4.5), sharey=True)
    colors = {"symbolic": "tab:blue", "noun_question": "tab:orange", "verb_instruction": "tab:green"}
    for axis, role in zip(axes, POSITIONS):
        for style in STYLES:
            selected = [r for r in result_rows if r["token_position_role"] == role and r["condition"] == "heldout_operands_" + style]
            axis.plot([r["layer_index"] for r in selected], [r["accuracy"] for r in selected], marker=".", label=style.replace("_", " "), color=colors[style])
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance (50%)")
        axis.set(title=role.replace("_", " "), xlabel="Hidden-state index", ylim=(0, 1.02))
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Operation classification accuracy")
    axes[-1].legend(fontsize="small")
    fig.suptitle("Addition vs. multiplication decoding by token role")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)

    print(f"train pairs: {len(train_pairs)}; held-out pairs: {len(test_pairs)}")
    print("operation cue uses the last tokenizer token overlapping its labeled character span")
    print("probe position roles: " + ", ".join(POSITIONS))
    print(f"saved results: {results_path}")
    print(f"saved examples: {examples_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
