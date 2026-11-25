"""Compare final-position and mean-pooled states for controlled prompt pairs."""

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
REFERENCE_ID = "multiply_anchor"

# All comparisons use the same arithmetic target. The hard negatives preserve
# words or structure while changing the required computation; unrelated items
# provide a low-overlap baseline.
PROMPTS = [
    {
        "id": "multiply_anchor",
        "category": "equivalent",
        "relation_to_anchor": "reference",
        "text": "Calculate 17 × 23.",
        "expected_answer": "391",
    },
    {
        "id": "multiply_minor_wording",
        "category": "equivalent",
        "relation_to_anchor": "minor_wording_change",
        "text": "Compute 17 × 23.",
        "expected_answer": "391",
    },
    {
        "id": "multiply_product_paraphrase",
        "category": "equivalent",
        "relation_to_anchor": "paraphrase",
        "text": "What is the product of 17 and 23?",
        "expected_answer": "391",
    },
    {
        "id": "multiply_reversed_order",
        "category": "equivalent",
        "relation_to_anchor": "reordered_operands",
        "text": "Multiply 23 by 17.",
        "expected_answer": "391",
    },
    {
        "id": "multiply_story",
        "category": "equivalent",
        "relation_to_anchor": "story_framing",
        "text": (
            "Alice has 17 boxes containing 23 widgets each. "
            "How many widgets does she have?"
        ),
        "expected_answer": "391",
    },
    {
        "id": "add_same_operands",
        "category": "hard_negative",
        "relation_to_anchor": "same_operands_different_operation",
        "text": "Calculate 17 + 23.",
        "expected_answer": "40",
    },
    {
        "id": "multiply_changed_operand",
        "category": "hard_negative",
        "relation_to_anchor": "same_operation_changed_operand",
        "text": "Calculate 17 × 24.",
        "expected_answer": "408",
    },
    {
        "id": "capital_of_japan",
        "category": "unrelated",
        "relation_to_anchor": "unrelated_factual_question",
        "text": "What is the capital of Japan?",
        "expected_answer": "Tokyo",
    },
    {
        "id": "days_in_week",
        "category": "unrelated",
        "relation_to_anchor": "unrelated_counting_question",
        "text": "How many days are in a week?",
        "expected_answer": "7",
    },
]

# Compare every other prompt with the same multiplication reference.
PAIR_DEFINITIONS = [
    ("multiply_minor_wording", "equivalent", "minor_wording_change"),
    ("multiply_product_paraphrase", "equivalent", "paraphrase"),
    ("multiply_reversed_order", "equivalent", "reordered_operands"),
    ("multiply_story", "equivalent", "story_framing"),
    (
        "add_same_operands",
        "hard_negative",
        "same_operands_different_operation",
    ),
    (
        "multiply_changed_operand",
        "hard_negative",
        "same_operation_changed_operand",
    ),
    ("capital_of_japan", "unrelated", "unrelated_factual_question"),
    ("days_in_week", "unrelated", "unrelated_counting_question"),
]


def unique_output_paths(results_dir):
    """Return a set of output paths that does not overwrite a prior run."""
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
        }
        if not any(path.exists() for path in paths.values()):
            return paths
        suffix += 1


def token_jaccard(tokenizer, text_a, text_b):
    """Measure lexical overlap from unique prompt-body token IDs."""
    tokens_a = set(
        tokenizer(text_a, add_special_tokens=False)["input_ids"]
    )
    tokens_b = set(
        tokenizer(text_b, add_special_tokens=False)["input_ids"]
    )
    union = tokens_a | tokens_b
    return len(tokens_a & tokens_b) / len(union) if union else 1.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"device: {device}; model dtype: {dtype}")

    # Use the already selected model, and fail clearly rather than downloading
    # files if it is not present in the local Hugging Face cache.
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

    hidden_states_by_prompt = {}
    token_count_by_prompt = {}

    for prompt in PROMPTS:
        model_text = prompt["text"] + ANSWER_SUFFIX
        inputs = tokenizer(model_text, return_tensors="pt").to(device)
        token_count_by_prompt[prompt["id"]] = inputs["input_ids"].shape[1]

        # The shared Answer: suffix gives every example the same final input
        # position. Mean pooling also summarizes all real (non-padding) tokens.
        with torch.inference_mode():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        token_mask = inputs["attention_mask"][0].bool()
        hidden_states_by_prompt[prompt["id"]] = {
            "final_answer_position": [
                state[0, -1, :].detach().float().cpu()
                for state in outputs.hidden_states
            ],
            "mean_pool_non_padding": [
                state[0, token_mask, :].mean(dim=0).detach().float().cpu()
                for state in outputs.hidden_states
            ],
        }
        print(
            f"{prompt['id']}: "
            f"{token_count_by_prompt[prompt['id']]} model-input tokens"
        )
        del outputs

    reference = next(
        prompt for prompt in PROMPTS if prompt["id"] == REFERENCE_ID
    )
    prompt_by_id = {prompt["id"]: prompt for prompt in PROMPTS}
    num_hidden_state_layers = len(
        hidden_states_by_prompt[REFERENCE_ID]["final_answer_position"]
    )
    pair_rows = []

    for other_id, category, relation in PAIR_DEFINITIONS:
        other = prompt_by_id[other_id]
        overlap = token_jaccard(tokenizer, reference["text"], other["text"])

        for representation in (
            "final_answer_position",
            "mean_pool_non_padding",
        ):
            for layer_index in range(num_hidden_state_layers):
                vector_a = hidden_states_by_prompt[REFERENCE_ID][representation][
                    layer_index
                ]
                vector_b = hidden_states_by_prompt[other_id][representation][
                    layer_index
                ]
                cosine = F.cosine_similarity(vector_a, vector_b, dim=0).item()
                layer_name = (
                    "embedding_output"
                    if layer_index == 0
                    else f"transformer_layer_{layer_index}"
                )
                pair_rows.append(
                    {
                        "prompt_a": REFERENCE_ID,
                        "prompt_b": other_id,
                        "model_name": MODEL_NAME,
                        "device": device,
                        "model_dtype": str(model_dtype),
                        "representation": representation,
                        "similarity_metric": "cosine",
                        "category": category,
                        "relation": relation,
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "cosine_similarity": cosine,
                        "body_token_jaccard": overlap,
                        "prompt_a_token_count": token_count_by_prompt[REFERENCE_ID],
                        "prompt_b_token_count": token_count_by_prompt[other_id],
                    }
                )

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_paths = unique_output_paths(results_dir)

    with output_paths["prompts"].open(
        "w", newline="", encoding="utf-8"
    ) as prompt_file:
        fields = [
            "id",
            "category",
            "relation_to_anchor",
            "text",
            "appended_suffix",
            "expected_answer",
            "model_name",
            "device",
            "model_dtype",
        ]
        writer = csv.DictWriter(prompt_file, fieldnames=fields)
        writer.writeheader()
        for prompt in PROMPTS:
            writer.writerow(
                {
                    **prompt,
                    "appended_suffix": ANSWER_SUFFIX,
                    "model_name": MODEL_NAME,
                    "device": device,
                    "model_dtype": str(model_dtype),
                }
            )

    with output_paths["pairs"].open(
        "w", newline="", encoding="utf-8"
    ) as pair_file:
        writer = csv.DictWriter(pair_file, fieldnames=pair_rows[0].keys())
        writer.writeheader()
        writer.writerows(pair_rows)

    layer_indices = list(range(num_hidden_state_layers))
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, representation, title in zip(
        axes,
        ("final_answer_position", "mean_pool_non_padding"),
        ("Final Answer: position", "Mean pooled token states"),
    ):
        for category in ("equivalent", "hard_negative", "unrelated"):
            means = []
            for layer_index in layer_indices:
                values = [
                    row["cosine_similarity"]
                    for row in pair_rows
                    if row["representation"] == representation
                    and row["category"] == category
                    and row["layer_index"] == layer_index
                ]
                means.append(statistics.mean(values))
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

    axes[0].set_ylabel("Cosine similarity to multiplication reference")
    figure.suptitle("Prompt similarity by representation method and layer")
    figure.tight_layout()
    figure.savefig(output_paths["layers_plot"], dpi=160)
    plt.close(figure)

    final_layer = num_hidden_state_layers - 1
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    colors = {
        "equivalent": "tab:blue",
        "hard_negative": "tab:orange",
        "unrelated": "tab:green",
    }
    final_rows = [
        row for row in pair_rows if row["layer_index"] == final_layer
    ]
    for axis, representation, title in zip(
        axes,
        ("final_answer_position", "mean_pool_non_padding"),
        ("Final Answer: position", "Mean pooled token states"),
    ):
        representation_rows = [
            row
            for row in final_rows
            if row["representation"] == representation
        ]
        for category, color in colors.items():
            selected = [
                row for row in representation_rows if row["category"] == category
            ]
            axis.scatter(
                [row["body_token_jaccard"] for row in selected],
                [row["cosine_similarity"] for row in selected],
                label=category.replace("_", " "),
                color=color,
                s=55,
            )
            for row in selected:
                axis.annotate(
                    row["prompt_b"],
                    (row["body_token_jaccard"], row["cosine_similarity"]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                )

        axis.set_title(title)
        axis.set_xlabel("Prompt-body token Jaccard overlap with reference")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[0].set_ylabel("Final-layer cosine similarity")
    figure.suptitle("Final-layer similarity vs. lexical overlap")
    figure.tight_layout()
    figure.savefig(output_paths["overlap_plot"], dpi=160)
    plt.close(figure)

    print(f"hidden-state layers including embedding output: {num_hidden_state_layers}")
    for key, path in output_paths.items():
        print(f"saved {key}: {path}")


if __name__ == "__main__":
    main()
