"""Expand semantic-reuse comparisons across arithmetic targets and stories."""

import csv
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
UNRELATED_PROMPTS = [
    ("unrelated_capital_japan", "What is the capital of Japan?", "Tokyo"),
    ("unrelated_days_week", "How many days are in a week?", "7"),
]

# Each target has two equivalent variants and two hard negatives. Together
# with two unrelated controls, each target contributes a balanced 2/2/2 set.
TARGETS = [
    {
        "id": "multiply_17_23",
        "operation": "multiplication",
        "reference": ("Calculate 17 × 23.", "17 × 23", "391"),
        "equivalent": [
            ("Compute 17 × 23.", "391"),
            ("What is the product of 17 and 23?", "391"),
        ],
        "hard_negative": [
            ("Calculate 17 + 23.", "40"),
            ("Calculate 17 × 24.", "408"),
        ],
    },
    {
        "id": "add_12_19",
        "operation": "addition",
        "reference": ("Calculate 12 + 19.", "12 + 19", "31"),
        "equivalent": [
            ("Compute 12 + 19.", "31"),
            ("What is the sum of 12 and 19?", "31"),
        ],
        "hard_negative": [
            ("Calculate 12 × 19.", "228"),
            ("Calculate 12 + 20.", "32"),
        ],
    },
    {
        "id": "subtract_46_18",
        "operation": "subtraction",
        "reference": ("Calculate 46 − 18.", "46 − 18", "28"),
        "equivalent": [
            ("Compute 46 − 18.", "28"),
            ("What is 18 subtracted from 46?", "28"),
        ],
        "hard_negative": [
            ("Calculate 46 + 18.", "64"),
            ("Calculate 46 − 19.", "27"),
        ],
    },
    {
        "id": "multiply_14_16",
        "operation": "multiplication",
        "reference": ("Calculate 14 × 16.", "14 × 16", "224"),
        "equivalent": [
            ("Compute 14 × 16.", "224"),
            ("What is the product of 14 and 16?", "224"),
        ],
        "hard_negative": [
            ("Calculate 14 + 16.", "30"),
            ("Calculate 14 × 17.", "238"),
        ],
    },
]

# These prompts preserve the 17 × 23 subproblem while changing its framing or
# embedding it in a larger computation. Span comparisons localize that phrase;
# full-prompt comparisons include all surrounding context and the final query.
STORY_PROMPTS = [
    {
        "id": "multiply_17_23_story",
        "text": (
            "Alice has 17 boxes containing 23 widgets each. "
            "How many widgets does she have?"
        ),
        "computation_span": "17 boxes containing 23 widgets each",
        "expected_answer": "391",
        "category": "equivalent",
        "relation": "story_framing_same_computation",
    },
    {
        "id": "multiply_17_23_plus_12",
        "text": (
            "Alice has 17 boxes containing 23 widgets each and receives "
            "12 more widgets. How many widgets does she have in total?"
        ),
        "computation_span": "17 boxes containing 23 widgets each",
        "expected_answer": "403",
        "category": "embedded_full_prompt",
        "relation": "same_multiplication_then_add_12",
    },
    {
        "id": "multiply_17_23_minus_12",
        "text": (
            "Alice has 17 boxes containing 23 widgets each. She gives away "
            "12 widgets. How many does she have left?"
        ),
        "computation_span": "17 boxes containing 23 widgets each",
        "expected_answer": "379",
        "category": "embedded_full_prompt",
        "relation": "same_multiplication_then_subtract_12",
    },
]


def build_prompt_set_and_pairs():
    prompts = []
    pairs = []

    for target in TARGETS:
        target_id = target["id"]
        reference_text, reference_span, reference_answer = target["reference"]
        reference_id = f"{target_id}_reference"
        prompts.append(
            {
                "id": reference_id,
                "target_id": target_id,
                "role": "reference",
                "category": "reference",
                "relation": "canonical_arithmetic_prompt",
                "text": reference_text,
                "computation_span": reference_span,
                "expected_answer": reference_answer,
            }
        )

        for category in ("equivalent", "hard_negative"):
            for index, (text, answer) in enumerate(target[category], start=1):
                prompt_id = f"{target_id}_{category}_{index}"
                relation = (
                    "equivalent_variant"
                    if category == "equivalent"
                    else "changed_operation_or_operand"
                )
                prompts.append(
                    {
                        "id": prompt_id,
                        "target_id": target_id,
                        "role": category,
                        "category": category,
                        "relation": relation,
                        "text": text,
                        "computation_span": "",
                        "expected_answer": answer,
                    }
                )
                pairs.append(
                    {
                        "prompt_a": reference_id,
                        "prompt_b": prompt_id,
                        "target_id": target_id,
                        "scope": "full_prompt",
                        "category": category,
                        "relation": relation,
                    }
                )

        for unrelated_id, text, answer in UNRELATED_PROMPTS:
            if not any(prompt["id"] == unrelated_id for prompt in prompts):
                prompts.append(
                    {
                        "id": unrelated_id,
                        "target_id": "",
                        "role": "unrelated_control",
                        "category": "unrelated",
                        "relation": "unrelated_baseline",
                        "text": text,
                        "computation_span": "",
                        "expected_answer": answer,
                    }
                )
            pairs.append(
                {
                    "prompt_a": reference_id,
                    "prompt_b": unrelated_id,
                    "target_id": target_id,
                    "scope": "full_prompt",
                    "category": "unrelated",
                    "relation": "unrelated_baseline",
                }
            )

    for story in STORY_PROMPTS:
        prompts.append(
            {
                "id": story["id"],
                "target_id": "multiply_17_23",
                "role": "story_or_embedded_problem",
                "category": story["category"],
                "relation": story["relation"],
                "text": story["text"],
                "computation_span": story["computation_span"],
                "expected_answer": story["expected_answer"],
            }
        )
        pairs.append(
            {
                "prompt_a": "multiply_17_23_reference",
                "prompt_b": story["id"],
                "target_id": "multiply_17_23",
                "scope": "full_prompt",
                "category": story["category"],
                "relation": story["relation"],
            }
        )
        pairs.append(
            {
                "prompt_a": "multiply_17_23_reference",
                "prompt_b": story["id"],
                "target_id": "multiply_17_23",
                "scope": "computation_span",
                "category": "same_subcomputation",
                "relation": story["relation"],
            }
        )

    return prompts, pairs


def unique_output_paths(results_dir):
    """Return concise paths without overwriting an earlier experiment run."""
    suffix = 1
    while True:
        run_suffix = "" if suffix == 1 else f"_{suffix}"
        paths = {
            "prompts": results_dir / f"exp02_prompts{run_suffix}.csv",
            "pairs": results_dir
            / f"exp02_pairwise_similarity{run_suffix}.csv",
            "layers_plot": results_dir
            / f"exp02_layer_similarity{run_suffix}.png",
            "overlap_plot": results_dir
            / f"exp02_similarity_vs_lexical_overlap{run_suffix}.png",
            "span_plot": results_dir
            / f"exp02_subproblem_span_similarity{run_suffix}.png",
        }
        if not any(path.exists() for path in paths.values()):
            return paths
        suffix += 1


def token_jaccard(tokenizer, text_a, text_b):
    """Measure overlap between unique token IDs in two text spans."""
    tokens_a = set(
        tokenizer(text_a, add_special_tokens=False)["input_ids"]
    )
    tokens_b = set(
        tokenizer(text_b, add_special_tokens=False)["input_ids"]
    )
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 1.0


def token_indices_for_span(offsets, text, span):
    """Map an exact character span in a prompt to its tokenizer positions."""
    start = text.find(span)
    if start < 0:
        raise ValueError(f"Could not find computation span {span!r} in {text!r}.")
    end = start + len(span)
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start
        and token_start < end
        and token_end > start
    ]
    if not indices:
        raise ValueError(f"Computation span {span!r} matched no tokenizer tokens.")
    return indices


def main():
    prompts, pair_definitions = build_prompt_set_and_pairs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device: {device}; model dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError(
            "Computation-span comparisons require a fast tokenizer with offsets."
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    model.eval()
    model_dtype = next(model.parameters()).dtype

    # Keep three prompt-level representations at every layer. The span vector
    # is present only for prompts with a specifically annotated computation.
    vectors_by_prompt = {}
    token_count_by_prompt = {}
    for prompt in prompts:
        model_text = prompt["text"] + ANSWER_SUFFIX
        encoded = tokenizer(
            model_text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = encoded["offset_mapping"][0].tolist()
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key != "offset_mapping"
        }
        token_mask = model_inputs["attention_mask"][0].bool()
        token_count_by_prompt[prompt["id"]] = int(token_mask.sum().item())

        span_indices = None
        if prompt["computation_span"]:
            # Qwen is causal: states inside the span include preceding text but
            # cannot incorporate the question or operations that follow it.
            span_indices = token_indices_for_span(
                offsets,
                model_text,
                prompt["computation_span"],
            )

        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        prompt_vectors = {
            "final_answer_position": [],
            "mean_pool_non_padding": [],
            "mean_pool_computation_span": [],
        }
        for state in outputs.hidden_states:
            prompt_vectors["final_answer_position"].append(
                state[0, -1, :].detach().float().cpu()
            )
            prompt_vectors["mean_pool_non_padding"].append(
                state[0, token_mask, :].mean(dim=0).detach().float().cpu()
            )
            if span_indices is not None:
                prompt_vectors["mean_pool_computation_span"].append(
                    state[0, span_indices, :].mean(dim=0).detach().float().cpu()
                )
        vectors_by_prompt[prompt["id"]] = prompt_vectors
        print(
            f"{prompt['id']}: {token_count_by_prompt[prompt['id']]} model-input "
            "tokens"
        )
        del outputs

    num_hidden_state_layers = len(
        vectors_by_prompt["multiply_17_23_reference"][
            "final_answer_position"
        ]
    )
    prompt_by_id = {prompt["id"]: prompt for prompt in prompts}
    pair_rows = []

    for pair in pair_definitions:
        prompt_a = prompt_by_id[pair["prompt_a"]]
        prompt_b = prompt_by_id[pair["prompt_b"]]
        is_span_comparison = pair["scope"] == "computation_span"
        if is_span_comparison:
            text_a = prompt_a["computation_span"]
            text_b = prompt_b["computation_span"]
            methods = ("mean_pool_computation_span",)
            overlap_scope = "computation_span"
        else:
            text_a = prompt_a["text"]
            text_b = prompt_b["text"]
            methods = (
                "final_answer_position",
                "mean_pool_non_padding",
            )
            overlap_scope = "full_prompt_body"
        overlap = token_jaccard(tokenizer, text_a, text_b)

        for method in methods:
            for layer_index in range(num_hidden_state_layers):
                vector_a = vectors_by_prompt[pair["prompt_a"]][method][
                    layer_index
                ]
                vector_b = vectors_by_prompt[pair["prompt_b"]][method][
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
                        "prompt_a": pair["prompt_a"],
                        "prompt_b": pair["prompt_b"],
                        "target_id": pair["target_id"],
                        "comparison_scope": pair["scope"],
                        "category": pair["category"],
                        "relation": pair["relation"],
                        "representation": method,
                        "similarity_metric": "cosine",
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "cosine_similarity": F.cosine_similarity(
                            vector_a,
                            vector_b,
                            dim=0,
                        ).item(),
                        "lexical_overlap_jaccard": overlap,
                        "lexical_overlap_scope": overlap_scope,
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
        "target_id",
        "role",
        "category",
        "relation",
        "text",
        "appended_suffix",
        "computation_span",
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
    full_prompt_rows = [
        row for row in pair_rows if row["comparison_scope"] == "full_prompt"
    ]
    full_categories = (
        "equivalent",
        "hard_negative",
        "unrelated",
        "embedded_full_prompt",
    )
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for axis, method, title in zip(
        axes,
        ("final_answer_position", "mean_pool_non_padding"),
        ("Final Answer: position", "Mean pooled full prompt"),
    ):
        for category in full_categories:
            means = []
            for layer_index in layer_indices:
                values = [
                    row["cosine_similarity"]
                    for row in full_prompt_rows
                    if row["representation"] == method
                    and row["category"] == category
                    and row["layer_index"] == layer_index
                ]
                if values:
                    means.append(statistics.mean(values))
                else:
                    means.append(float("nan"))
            axis.plot(
                layer_indices,
                means,
                marker=".",
                label=category.replace("_", " "),
            )
        axis.set_title(title)
        axis.set_xlabel("Hidden-state index (0 = embedding output)")
        axis.set_xticks(layer_indices)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Cosine similarity to each target reference")
    figure.suptitle("Full-prompt similarity across arithmetic targets and layers")
    figure.tight_layout()
    figure.savefig(output_paths["layers_plot"], dpi=160)
    plt.close(figure)

    final_layer = num_hidden_state_layers - 1
    final_rows = [
        row
        for row in full_prompt_rows
        if row["layer_index"] == final_layer
    ]
    colors = {
        "equivalent": "tab:blue",
        "hard_negative": "tab:orange",
        "unrelated": "tab:green",
        "embedded_full_prompt": "tab:red",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
    for axis, method, title in zip(
        axes,
        ("final_answer_position", "mean_pool_non_padding"),
        ("Final Answer: position", "Mean pooled full prompt"),
    ):
        method_rows = [
            row for row in final_rows if row["representation"] == method
        ]
        for category, color in colors.items():
            selected = [
                row for row in method_rows if row["category"] == category
            ]
            if not selected:
                continue
            axis.scatter(
                [row["lexical_overlap_jaccard"] for row in selected],
                [row["cosine_similarity"] for row in selected],
                label=category.replace("_", " "),
                color=color,
                s=45,
            )
        axis.set_title(title)
        axis.set_xlabel("Prompt-body token Jaccard overlap")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[0].set_ylabel("Final-layer cosine similarity")
    figure.suptitle("Full-prompt similarity vs. lexical overlap")
    figure.tight_layout()
    figure.savefig(output_paths["overlap_plot"], dpi=160)
    plt.close(figure)

    span_rows = [
        row for row in pair_rows if row["comparison_scope"] == "computation_span"
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    for row_prompt_id in (
        "multiply_17_23_story",
        "multiply_17_23_plus_12",
        "multiply_17_23_minus_12",
    ):
        values = [
            row
            for row in span_rows
            if row["prompt_b"] == row_prompt_id
            and row["representation"] == "mean_pool_computation_span"
        ]
        axis.plot(
            [row["layer_index"] for row in values],
            [row["cosine_similarity"] for row in values],
            marker=".",
            label=row_prompt_id.replace("multiply_17_23_", "").replace("_", " "),
        )
    axis.set_xlabel("Hidden-state index (0 = embedding output)")
    axis.set_ylabel("Cosine similarity between 17 × 23 spans")
    axis.set_title("Local multiplication-span similarity across layers")
    axis.set_xticks(layer_indices)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_paths["span_plot"], dpi=160)
    plt.close(figure)

    print(
        f"prompt bodies: {len(prompts)}; pair definitions: {len(pair_definitions)}; "
        f"pairwise rows: {len(pair_rows)}; hidden-state indices: "
        f"{num_hidden_state_layers}"
    )
    for key, path in output_paths.items():
        print(f"saved {key}: {path}")


if __name__ == "__main__":
    main()
