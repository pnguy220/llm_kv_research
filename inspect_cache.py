import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face model that we want to inspect.
# Qwen2.5-0.5B is a small decoder-only transformer model.
MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def tensor_bytes(tensor):
    """Return the bytes occupied by a tensor's elements.

    This measures the K/V tensor data itself, not PyTorch allocator overhead or
    total CUDA memory use.
    """
    return tensor.numel() * tensor.element_size()


def kv_cache_bytes(cache):
    """Return total bytes stored by all cached K and V tensors."""
    total_bytes = 0

    for layer in cache.layers:
        total_bytes += tensor_bytes(layer.keys)
        total_bytes += tensor_bytes(layer.values)

    return total_bytes


def unique_result_paths(results_dir):
    """Choose CSV and plot paths without replacing an earlier experiment."""
    suffix = 1

    while True:
        run_suffix = "" if suffix == 1 else f"_{suffix}"
        csv_path = results_dir / f"exp01_kv_cache_context{run_suffix}.csv"
        plot_path = results_dir / f"exp01_kv_cache_context{run_suffix}.png"

        if not csv_path.exists() and not plot_path.exists():
            return csv_path, plot_path

        suffix += 1


# Use the NVIDIA GPU through CUDA if PyTorch can detect it.
# Otherwise, fall back to running on the CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Use float16 on the GPU to reduce model and KV-cache memory usage.
# On CPU, use float32 for better compatibility.
dtype = torch.float16 if device == "cuda" else torch.float32

print("device:", device)
print("dtype:", dtype)


# Load the tokenizer associated with the model.
# The tokenizer converts text into integer token IDs that the model understands.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# Load the pretrained language model.
# dtype controls the numerical representation of the model's tensors.
# .to(device) moves the model onto either the GPU or CPU selected above.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
).to(device)


# Put the model into evaluation mode.
# This disables training-specific behavior such as dropout.
model.eval()


# Derive the expected K/V storage for one token from the loaded model config.
# A cache stores both K and V for each KV head in every transformer layer.
config = model.config
num_layers = config.num_hidden_layers
num_attention_heads = config.num_attention_heads
num_kv_heads = getattr(config, "num_key_value_heads", None)
if num_kv_heads is None:
    # For models without grouped-query attention, each attention head has a KV.
    num_kv_heads = num_attention_heads

head_dim = getattr(config, "head_dim", None)
if head_dim is None:
    head_dim = config.hidden_size // num_attention_heads

model_dtype = next(model.parameters()).dtype
bytes_per_element = next(model.parameters()).element_size()
expected_bytes_per_token = (
    num_layers * 2 * num_kv_heads * head_dim * bytes_per_element
)
print(
    "configuration-derived KV bytes per token: "
    f"{expected_bytes_per_token:,} bytes "
    f"({expected_bytes_per_token / 1024:.2f} KiB) "
    f"[layers={num_layers}, KV heads={num_kv_heads}, head_dim={head_dim}, "
    f"dtype={model_dtype}]"
)


# Initial text that will be passed into the model.
# Processing this entire prompt is called the "prefill" stage.
#prompt = "The capital of France is"
prompt = "In the middle of May, the weather will be"


# Convert the prompt into token IDs.
# return_tensors="pt" tells the tokenizer to return PyTorch tensors.
# .to(device) moves those tensors onto the same device as the model.
inputs = tokenizer(prompt, return_tensors="pt").to(device)


# input_ids has shape:
# [batch_size, sequence_length]
#
# For this prompt we expect something like:
# [1, 5]
#
# meaning one prompt containing five tokens.
print("input_ids shape:", inputs["input_ids"].shape)
print("prompt tokens:", inputs["input_ids"].shape[1])


# Disable gradient calculation because we are only performing inference.
# inference_mode also avoids inference-only autograd bookkeeping.
with torch.inference_mode():

    # Run the initial prompt through the transformer.
    #
    # This is the PREFILL stage.
    #
    # use_cache=True tells the model to save the Key (K) and Value (V)
    # tensors produced by each attention layer so they can be reused
    # during autoregressive token generation.
    outputs = model(
        **inputs,
        use_cache=True,
    )


# Retrieve the KV cache produced during prefill.
#
# Hugging Face returns this as a DynamicCache object.
# It contains cached K and V tensors for every transformer layer.
cache = outputs.past_key_values

print("cache type:", type(cache))


# Sum the storage of every K and V tensor. This intentionally excludes model
# weights, temporary activations, and CUDA allocator overhead.
total_kv_bytes = kv_cache_bytes(cache)
print(
    "total KV-cache tensor memory: "
    f"{total_kv_bytes:,} bytes ({total_kv_bytes / 1024:.2f} KiB)"
)


# Inspect the KV cache for every transformer layer.
#
# Qwen2.5-0.5B has 24 transformer layers, so we expect
# this loop to print layers 0 through 23.
for i, layer in enumerate(cache.layers):

    # Cached Key tensor for this attention layer.
    k = layer.keys

    # Cached Value tensor for this attention layer.
    v = layer.values

    # For our 5-token prompt, we observed:
    #
    # K = (1, 2, 5, 64)
    # V = (1, 2, 5, 64)
    #
    # The dimensions represent:
    #
    # 1  = batch size
    # 2  = number of KV heads
    # 5  = sequence/context length (number of cached tokens)
    # 64 = dimension of each attention head
    #
    # dtype=torch.float16 means each stored element occupies 2 bytes.
    key_bytes = tensor_bytes(k)
    value_bytes = tensor_bytes(v)

    print(
        f"layer {i:2d}: "
        f"K={tuple(k.shape)} "
        f"V={tuple(v.shape)} "
        f"dtype={k.dtype} "
        f"K_bytes={key_bytes:,} "
        f"V_bytes={value_bytes:,} "
        f"KV_bytes={key_bytes + value_bytes:,}"
    )


# Compare the formula above with actual cache tensors for several prefixes of
# one fixed tokenized text. Each run is an independent prefill measurement.
context_lengths = (5, 10, 20, 40)
sweep_text = "The capital of France is Paris. " * 20
sweep_inputs = tokenizer(sweep_text, return_tensors="pt").to(device)
available_tokens = sweep_inputs["input_ids"].shape[1]
if available_tokens < max(context_lengths):
    raise ValueError(
        f"Context sweep needs {max(context_lengths)} tokens, but tokenization "
        f"produced only {available_tokens}."
    )

sweep_rows = []
for context_length in context_lengths:
    # Prefix slicing keeps the text identical while varying only its length.
    sweep_input_ids = sweep_inputs["input_ids"][:, :context_length]
    sweep_attention_mask = torch.ones_like(sweep_input_ids)

    with torch.inference_mode():
        sweep_outputs = model(
            input_ids=sweep_input_ids,
            attention_mask=sweep_attention_mask,
            use_cache=True,
        )

    sweep_cache = sweep_outputs.past_key_values
    actual_bytes = kv_cache_bytes(sweep_cache)
    batch_size = sweep_input_ids.shape[0]
    expected_bytes = (
        batch_size * context_length * expected_bytes_per_token
    )
    cache_dtype = sweep_cache.layers[0].keys.dtype
    row = {
        "context_tokens": context_length,
        "batch_size": batch_size,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "model_dtype": str(model_dtype),
        "cache_dtype": str(cache_dtype),
        "bytes_per_element": bytes_per_element,
        "expected_bytes_per_token": expected_bytes_per_token,
        "expected_kv_bytes": expected_bytes,
        "actual_kv_bytes": actual_bytes,
        "difference_bytes": actual_bytes - expected_bytes,
        "matches": actual_bytes == expected_bytes,
    }
    sweep_rows.append(row)

    print(
        f"prefill sweep: context={context_length:2d} tokens, "
        f"actual={actual_bytes:,} bytes, expected={expected_bytes:,} bytes, "
        f"difference={actual_bytes - expected_bytes:+,} bytes, "
        f"cache dtype={cache_dtype}"
    )

# Save a CSV and a plot with unique names so reruns preserve earlier results.
results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)
csv_path, plot_path = unique_result_paths(results_dir)

with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=sweep_rows[0].keys())
    writer.writeheader()
    writer.writerows(sweep_rows)

figure, axis = plt.subplots()
axis.plot(
    [row["context_tokens"] for row in sweep_rows],
    [row["actual_kv_bytes"] / 1024 for row in sweep_rows],
    marker="o",
    label="Measured K/V tensors",
)
axis.plot(
    [row["context_tokens"] for row in sweep_rows],
    [row["expected_kv_bytes"] / 1024 for row in sweep_rows],
    linestyle="--",
    label="Configuration estimate",
)
axis.set_xlabel("Context length (tokens)")
axis.set_ylabel("K/V tensor memory (KiB)")
axis.set_title("KV-cache memory vs. prefill context length")
axis.grid(True, alpha=0.3)
axis.legend()
figure.tight_layout()
figure.savefig(plot_path, dpi=150)
plt.close(figure)
print(f"saved context sweep CSV: {csv_path}")
print(f"saved context sweep plot: {plot_path}")


# Greedy autoregressive decoding: choose the highest-scoring next token,
# feed only that one token into the next forward pass, and reuse the cache.
NUM_NEW_TOKENS = 5
next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
generated_tokens = []

for step in range(NUM_NEW_TOKENS):
    # Save the token predicted by prefill (or the previous decode step).
    generated_tokens.append(next_token)

    # The next forward pass will consume one token, so its attention mask
    # covers the existing cached tokens plus that incoming token.
    cached_length = cache.layers[0].keys.shape[-2]
    decode_attention_mask = torch.ones(
        (inputs["input_ids"].shape[0], cached_length + 1),
        dtype=inputs["attention_mask"].dtype,
        device=device,
    )

    with torch.inference_mode():
        decode_outputs = model(
            input_ids=next_token,
            attention_mask=decode_attention_mask,
            past_key_values=cache,
            use_cache=True,
        )

    # The returned cache now includes the token fed into this decode pass.
    cache = decode_outputs.past_key_values
    cached_length = cache.layers[0].keys.shape[-2]
    total_kv_bytes = kv_cache_bytes(cache)
    print(
        f"decode step {step + 1}: "
        f"cached tokens={cached_length}, "
        f"KV-cache={total_kv_bytes:,} bytes "
        f"({total_kv_bytes / 1024:.2f} KiB)"
    )

    # These logits predict the token that will be fed in on the next step.
    next_token = decode_outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

# Decode the generated token IDs for a readable summary.
generated_ids = torch.cat(generated_tokens, dim=-1)[0]
print("generated text:", tokenizer.decode(generated_ids))
