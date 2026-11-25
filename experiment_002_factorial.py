"""Controlled layerwise comparisons of arithmetic prompt factors.

This run requires CUDA and FP16. It builds matched prompt pairs that differ
in exactly one factor: operation, first operand, second operand, wording, or
symbolic-versus-word operator form.
"""

import csv
from itertools import combinations
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
ANSWER_SUFFIX = "\nAnswer:"
OPERANDS_A = (24, 36)
OPERANDS_B = (3, 6)
WORDINGS = ("calculate", "compute", "question")
SURFACE_FORMS = ("symbol", "word")

OPERATIONS = {
    "add": {"symbol": "+", "word": "plus"},
    "subtract": {"symbol": "−", "word": "minus"},
    "multiply": {"symbol": "×", "word": "times"},
    "divide": {"symbol": "÷", "word": "divided by"},
}
FACTOR_ORDER = (
    "operation",
    "operand_a",
    "operand_b",
    "wording",
    "surface_form",
)


def render_operator(operation, surface_form):
    return OPERATIONS[operation][surface_form]


def render_expression(operation, operand_a, operand_b, surface_form):
    operator = render_operator(operation, surface_form)
    return f"{operand_a} {operator} {operand_b}"


def render_prompt(operation, operand_a, operand_b, wording, surface_form):
    expression = render_expression(
        operation,
        operand_a,
        operand_b,
        surface_form,
    )
    if wording == "calculate":
        return f"Calculate {expression}."
    if wording == "compute":
        return f"Compute {expression}."
    if wording == "question":
        return f"What is {expression}?"
    raise ValueError(f"Unknown wording template: {wording}")


def expected_answer(operation, operand_a, operand_b):
    if operation == "add":
        return operand_a + operand_b
    if operation == "subtract":
        return operand_a - operand_b
    if operation == "multiply":
        return operand_a * operand_b
    if operation == "divide":
        return operand_a // operand_b
    raise ValueError(f"Unknown operation: {operation}")


def prompt_id(operation, operand_a, operand_b, wording, surface_form):
    return (
        f"{operation}_{operand_a}_{operand_b}_"
        f"{wording}_{surface_form}"
    )


def build_prompt_grid_and_pairs():
    prompts = []
    prompt_ids = {}

    for operation in OPERATIONS:
        for operand_a in OPERANDS_A:
            for operand_b in OPERANDS_B:
                for wording in WORDINGS:
                    for surface_form in SURFACE_FORMS:
                        current_id = prompt_id(
                            operation,
                            operand_a,
                            operand_b,
                            wording,
                            surface_form,
                        )
                        prompt_ids[
                            (operation, operand_a, operand_b, wording, surface_form)
                        ] = current_id
                        prompts.append(
                            {
                                "id": current_id,
                                "operation": operation,
                                "operand_a": operand_a,
                                "operand_b": operand_b,
                                "wording": wording,
                                "surface_form": surface_form,
                                "text": render_prompt(
                                    operation,
                                    operand_a,
                                    operand_b,
                                    wording,
                                    surface_form,
                                ),
                                "expected_answer": expected_answer(
                                    operation,
                                    operand_a,
                                    operand_b,
                                ),
                            }
                        )

    pairs = []

    def add_pair(factor, contrast, values_a, values_b):
        """Add a pair and verify exactly the named factor changes."""
        changed = [key for key in values_a if values_a[key] != values_b[key]]
        if changed != [factor]:
            raise ValueError(
                f"Pair intended to vary {factor}, but changes {changed}."
            )
        key_a = (
            values_a["operation"],
            values_a["operand_a"],
            values_a["operand_b"],
            values_a["wording"],
            values_a["surface_form"],
        )
        key_b = (
            values_b["operation"],
            values_b["operand_a"],
            values_b["operand_b"],
            values_b["wording"],
            values_b["surface_form"],
        )
        pairs.append(
            {
                "pair_id": f"pair_{len(pairs) + 1:04d}",
                "factor_changed": factor,
                "contrast": contrast,
                "prompt_a": prompt_ids[key_a],
                "prompt_b": prompt_ids[key_b],
                **{f"{key}_a": values_a[key] for key in values_a},
                **{f"{key}_b": values_b[key] for key in values_b},
            }
        )

    # Operation contrasts hold operands, wording, and operator style fixed.
    for operand_a in OPERANDS_A:
        for operand_b in OPERANDS_B:
            for wording in WORDINGS:
                for surface_form in SURFACE_FORMS:
                    for operation_a, operation_b in combinations(OPERATIONS, 2):
                        values_a = {
                            "operation": operation_a,
                            "operand_a": operand_a,
                            "operand_b": operand_b,
                            "wording": wording,
                            "surface_form": surface_form,
                        }
                        values_b = {**values_a, "operation": operation_b}
                        add_pair(
                            "operation",
                            f"{operation_a}_vs_{operation_b}",
                            values_a,
                            values_b,
                        )

    # Operand contrasts change one operand while keeping the other fixed.
    for operation in OPERATIONS:
        for operand_b in OPERANDS_B:
            for wording in WORDINGS:
                for surface_form in SURFACE_FORMS:
                    values_a = {
                        "operation": operation,
                        "operand_a": OPERANDS_A[0],
                        "operand_b": operand_b,
                        "wording": wording,
                        "surface_form": surface_form,
                    }
                    values_b = {**values_a, "operand_a": OPERANDS_A[1]}
                    add_pair(
                        "operand_a",
                        f"{OPERANDS_A[0]}_vs_{OPERANDS_A[1]}",
                        values_a,
                        values_b,
                    )

        for operand_a in OPERANDS_A:
            for wording in WORDINGS:
                for surface_form in SURFACE_FORMS:
                    values_a = {
                        "operation": operation,
                        "operand_a": operand_a,
                        "operand_b": OPERANDS_B[0],
                        "wording": wording,
                        "surface_form": surface_form,
                    }
                    values_b = {**values_a, "operand_b": OPERANDS_B[1]}
                    add_pair(
                        "operand_b",
                        f"{OPERANDS_B[0]}_vs_{OPERANDS_B[1]}",
                        values_a,
                        values_b,
                    )

    # Wording contrasts preserve the operation, operands, and surface form.
    for operation in OPERATIONS:
        for operand_a in OPERANDS_A:
            for operand_b in OPERANDS_B:
                for surface_form in SURFACE_FORMS:
                    for wording_a, wording_b in combinations(WORDINGS, 2):
                        values_a = {
                            "operation": operation,
                            "operand_a": operand_a,
                            "operand_b": operand_b,
                            "wording": wording_a,
                            "surface_form": surface_form,
                        }
                        values_b = {**values_a, "wording": wording_b}
                        add_pair(
                            "wording",
                            f"{wording_a}_vs_{wording_b}",
                            values_a,
                            values_b,
                        )

    # Surface-form contrasts preserve meaning, wording, and operand order.
    for operation in OPERATIONS:
        for operand_a in OPERANDS_A:
            for operand_b in OPERANDS_B:
                for wording in WORDINGS:
                    values_a = {
                        "operation": operation,
                        "operand_a": operand_a,
                        "operand_b": operand_b,
                        "wording": wording,
                        "surface_form": "symbol",
                    }
                    values_b = {**values_a, "surface_form": "word"}
                    add_pair(
                        "surface_form",
                        "symbol_vs_word",
                        values_a,
                        values_b,
                    )

    expected_pair_counts = {
        "operation": 144,
        "operand_a": 48,
        "operand_b": 48,
        "wording": 96,
        "surface_form": 48,
    }
    actual_pair_counts = {
        factor: sum(pair["factor_changed"] == factor for pair in pairs)
        for factor in FACTOR_ORDER
    }
    if actual_pair_counts != expected_pair_counts:
        raise RuntimeError(f"Unexpected pair counts: {actual_pair_counts}")

    return prompts, pairs


def token_jaccard(tokenizer, text_a, text_b):
    """Measure unique tokenizer-token overlap between prompt bodies."""
    tokens_a = set(
        tokenizer(text_a, add_special_tokens=False)["input_ids"]
    )
    tokens_b = set(
        tokenizer(text_b, add_special_tokens=False)["input_ids"]
    )
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 1.0


def unique_output_paths(results_dir):
    """Choose output names without overwriting previous experiment results."""
    suffix = 1
    while True:
        run_suffix = "" if suffix == 1 else f"_{suffix}"
        paths = {
            "prompts": results_dir / f"exp02_factorial_prompts{run_suffix}.csv",
            "pairs": results_dir / f"exp02_factorial_pairs{run_suffix}.csv",
            "layer_plot": results_dir
            / f"exp02_factorial_similarity_by_factor{run_suffix}.png",
            "final_plot": results_dir
            / f"exp02_factorial_final_similarity{run_suffix}.png",
        }
        if not any(path.exists() for path in paths.values()):
            return paths
        suffix += 1


def main():
    prompts, pairs = build_prompt_grid_and_pairs()

    # This controlled run is intended for the project's CUDA/FP16 setup.
    # Fail before model loading or writing results rather than silently making
    # a CPU/FP32 dataset that would need to be kept separate.
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Activate the CUDA-enabled environment and "
            "rerun; this experiment does not fall back to CPU."
        )

    device = "cuda"
    dtype = torch.float16
    print(f"device: {device}; model dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    prompt_by_id = {prompt["id"]: prompt for prompt in prompts}
    token_count_by_prompt = {}
    vectors_by_prompt = {}

    for prompt in prompts:
        model_text = prompt["text"] + ANSWER_SUFFIX
        inputs = tokenizer(model_text, return_tensors="pt").to(device)
        token_mask = inputs["attention_mask"][0].bool()
        token_count_by_prompt[prompt["id"]] = int(token_mask.sum().item())

        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        vectors_by_prompt[prompt["id"]] = {
            "final_answer_position": [
                state[0, -1, :].detach().float().cpu()
                for state in outputs.hidden_states
            ],
            "mean_pool_non_padding": [
                state[0, token_mask, :].mean(dim=0).detach().float().cpu()
                for state in outputs.hidden_states
            ],
        }
        del outputs

    num_hidden_state_layers = len(
        vectors_by_prompt[prompts[0]["id"]]["final_answer_position"]
    )
    pair_rows = []
    for pair in pairs:
        prompt_a = prompt_by_id[pair["prompt_a"]]
        prompt_b = prompt_by_id[pair["prompt_b"]]
        overlap = token_jaccard(tokenizer, prompt_a["text"], prompt_b["text"])

        for representation in (
            "final_answer_position",
            "mean_pool_non_padding",
        ):
            for layer_index in range(num_hidden_state_layers):
                vector_a = vectors_by_prompt[pair["prompt_a"]][representation][
                    layer_index
                ]
                vector_b = vectors_by_prompt[pair["prompt_b"]][representation][
                    layer_index
                ]
                layer_name = (
                    "embedding_output"
                    if layer_index == 0
                    else f"transformer_layer_{layer_index}"
                )
                pair_rows.append(
                    {
                        "model_name": MODEL_NAME,
                        "device": device,
                        "model_dtype": str(model_dtype),
                        "pair_id": pair["pair_id"],
                        "factor_changed": pair["factor_changed"],
                        "contrast": pair["contrast"],
                        "prompt_a": pair["prompt_a"],
                        "prompt_b": pair["prompt_b"],
                        "operation_a": pair["operation_a"],
                        "operation_b": pair["operation_b"],
                        "operand_a_a": pair["operand_a_a"],
                        "operand_a_b": pair["operand_a_b"],
                        "operand_b_a": pair["operand_b_a"],
                        "operand_b_b": pair["operand_b_b"],
                        "wording_a": pair["wording_a"],
                        "wording_b": pair["wording_b"],
                        "surface_form_a": pair["surface_form_a"],
                        "surface_form_b": pair["surface_form_b"],
                        "representation": representation,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "cosine_similarity": F.cosine_similarity(
                            vector_a,
                            vector_b,
                            dim=0,
                        ).item(),
                        "body_token_jaccard": overlap,
                        "prompt_a_token_count": token_count_by_prompt[
                            pair["prompt_a"]
                        ],
                        "prompt_b_token_count": token_count_by_prompt[
                            pair["prompt_b"]
                        ],
                    }
                )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_paths = unique_output_paths(results_dir)

    prompt_fields = [
        "id",
        "operation",
        "operand_a",
        "operand_b",
        "wording",
        "surface_form",
        "text",
        "appended_suffix",
        "expected_answer",
    ]
    with output_paths["prompts"].open(
        "w", newline="", encoding="utf-8"
    ) as prompt_file:
        writer = csv.DictWriter(prompt_file, fieldnames=prompt_fields)
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({**prompt, "appended_suffix": ANSWER_SUFFIX})

    with output_paths["pairs"].open(
        "w", newline="", encoding="utf-8"
    ) as pair_file:
        writer = csv.DictWriter(pair_file, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    layer_indices = list(range(num_hidden_state_layers))
    colors = {
        "operation": "tab:blue",
        "operand_a": "tab:orange",
        "operand_b": "tab:green",
        "wording": "tab:red",
        "surface_form": "tab:purple",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, representation, title in zip(
        axes,
        ("final_answer_position", "mean_pool_non_padding"),
        ("Final Answer: position", "Mean pooled prompt"),
    ):
        for factor in FACTOR_ORDER:
            means = []
            for layer_index in layer_indices:
                values = [
                    row["cosine_similarity"]
                    for row in pair_rows
                    if row["representation"] == representation
                    and row["factor_changed"] == factor
                    and row["layer_index"] == layer_index
                ]
                means.append(statistics.mean(values))
            axis.plot(
                layer_indices,
                means,
                marker=".",
                label=factor.replace("_", " "),
                color=colors[factor],
            )
        axis.set_title(title)
        axis.set_xlabel("Hidden-state index (0 = embedding output)")
        axis.set_xticks(layer_indices)
        axis.set_ylim(0, 1.02)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Cosine similarity for pairs changing one factor")
    figure.suptitle("Layerwise similarity by controlled prompt factor")
    figure.tight_layout()
    figure.savefig(output_paths["layer_plot"], dpi=160)
    plt.close(figure)

    final_layer = num_hidden_state_layers - 1
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, representation, title in zip(
        axes,
        ("final_answer_position", "mean_pool_non_padding"),
        ("Final Answer: position", "Mean pooled prompt"),
    ):
        distributions = [
            [
                row["cosine_similarity"]
                for row in pair_rows
                if row["representation"] == representation
                and row["factor_changed"] == factor
                and row["layer_index"] == final_layer
            ]
            for factor in FACTOR_ORDER
        ]
        boxplot = axis.boxplot(
            distributions,
            patch_artist=True,
            showfliers=False,
        )
        axis.set_xticks(range(1, len(FACTOR_ORDER) + 1))
        axis.set_xticklabels(
            [factor.replace("_", " ") for factor in FACTOR_ORDER]
        )
        for patch, factor in zip(boxplot["boxes"], FACTOR_ORDER):
            patch.set_facecolor(colors[factor])
            patch.set_alpha(0.45)
        axis.set_title(title)
        axis.set_ylabel("Final-layer cosine similarity")
        axis.set_ylim(0, 1.02)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(True, axis="y", alpha=0.3)
    figure.suptitle("Final-layer similarity distributions by changed factor")
    figure.tight_layout()
    figure.savefig(output_paths["final_plot"], dpi=160)
    plt.close(figure)

    print(
        f"prompts: {len(prompts)}; matched pairs: {len(pairs)}; "
        f"pair/layer/representation rows: {len(pair_rows)}; "
        f"hidden-state indices: {num_hidden_state_layers}"
    )
    for key, path in output_paths.items():
        print(f"saved {key}: {path}")


if __name__ == "__main__":
    main()
