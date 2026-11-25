# Experiment Summary

For detailed measurements and methods, see [PROJECT_NOTES.md](PROJECT_NOTES.md).

## The question behind the project

When a language model answers a question, it creates internal representations
as it processes each word or symbol. It stores attention-related Key and Value
tensors in a **KV cache** so it can reuse earlier context while generating the
answer. This project asks
whether the model can recognize when two prompts require the same kind of
computation, and whether that information could eventually be reused instead
of computed again.

The experiments use small addition and multiplication questions with
Qwen2.5-0.5B. They build up from measuring the cache, to reading internal
information, to changing internal information and observing the answer.

## Experiment 001 — What is stored in the KV cache?

**What I did:** Measured the Key and Value tensors created during **prefill**
(the initial pass that processes all prompt tokens) and watched the cache grow
during **decoding** (the step-by-step generation of answer tokens). I varied
prompt length and compared measured tensor bytes with the amount predicted
from the model configuration.

**What I found:** In the tested FP16 setup, the cache used 12 KiB for each
token in the context. Measured tensor sizes matched the amount predicted from
the model configuration. The cache grew linearly as the context got longer.

**What this tells me:** The cache's contents and memory growth are measurable
for this model. This establishes what later reuse experiments might try to
change.

## Experiment 002 — Do similar computations look similar inside the model?

**What I did:** Compared **hidden states** (the model's vector representation
of each token) for arithmetic prompts that
asked the same question in different words, prompts that looked similar but
asked for a different operation or used different numbers, and unrelated
prompts. I measured **cosine similarity**, a score for how closely two vectors
point in the same direction, at different model layers.

**What I found:** Similarity scores did not reliably separate equivalent
computations from similar-looking prompts that asked for something different.
Combining token vectors by averaging made many different prompts look almost
identical. The CPU and CUDA runs had very similar scores.

**What this tells me:** A simple similarity score is not enough to identify
when computations are equivalent.

## Experiment 003 — Can a small classifier read the operation or numbers?

**What I did:** Trained **linear probes** (small classifiers that read a vector
and predict a label) on hidden states to
identify addition or multiplication, and to identify the operands. Some tests
used number pairs that were not used to train the classifier. Other tests
changed the prompt wording.

**What I found:** The operation was easy to identify for unseen number pairs
when prompts used the same symbolic style. Recognition across different
wording depended on the phrase and the model layer. The numbers were also
readable in some settings, but operand results varied more across wording.
Comparing different token positions showed that the location of the vector
also affected what the classifier could read.

**What this tells me:** Operation and operand information can be read from
some internal vectors, but the signal is not equally clear across every layer,
wording, or token position. A classifier reading information does not prove
the model uses that signal to calculate.

## Experiment 004 — Does changing operator cache information affect the answer?

**What I did:** Replaced the operator token's Key and Value entries in the
KV cache with
entries from a prompt using the same operation or the other operation, then
measured the target answer. The target prompt after the operator was
reprocessed using the changed cache.

**What I found:** Same-operation substitutions mostly preserved the answer.
Different-operation substitutions reduced answer accuracy most when applied
across middle layers or all layers. They did not consistently make the model
produce the answer for the substituted operation.

**What this tells me:** Operator cache entries can affect the answer, but these
tests do not show that a useful computation can be cleanly substituted or
that doing so saves time or memory.

## Experiment 005 — Does changing hidden-state information affect the answer?

**What I did:** Used **activation patching**: replaced hidden-state vectors at
the operator token, the final
prompt token, or both, using vectors from prompts with the same or different
operations. I repeated this at different model layers.

**What I found:** Replacing the operator vector with one from the other
operation disrupted answers more in early layers. Replacing the final prompt
vector had stronger effects in later layers. Changing both positions caused
disruption in early and later layers, with partial recovery between them.
Different-operation changes often disrupted the answer rather than switching
it to the other operation's answer.

**What this tells me:** Internal vectors at these positions can influence
answer predictions, and the influence varies across layers. This is a
different intervention from changing the KV cache.

## Overall status

So far, the experiments show that operation information can be read from some
internal representations and that changing selected cache entries or hidden
vectors can influence the model's answer. They have **not** shown a reliable,
wording-independent representation that can be reused across different
queries, or a reduction in computation or memory from reuse.

All findings are limited to the prompts, model, and settings tested here.

## Short glossary

- **Token:** A piece of text the model processes, such as a word, number, or
  symbol. A prompt is represented as a sequence of tokens.
- **Prefill:** The model's first pass over the complete prompt. It creates the
  initial KV cache and contextual hidden states.
- **Decode / decoding:** Generating the response one token at a time, using
  the existing KV cache from the prompt and previously generated tokens.
- **KV cache:** Per-layer Key and Value tensors that store attention
  information for tokens already processed. This avoids recomputing all prior
  token information at every decode step.
- **Hidden state / activation:** A vector representing a token at a particular
  point in the model. In this summary, “activation patching” means replacing
  such a vector with one from another prompt while the model runs.
- **Layer:** One processing stage in the transformer. Hidden states and
  attention computations are updated as information passes through layers.
- **Linear probe:** A simple learned classifier used to test whether a label,
  such as “addition,” can be read from a hidden-state vector.
- **Cosine similarity:** A measure of the directional similarity of two
  vectors. A high score means the vectors point in similar directions; it
  does not by itself mean the prompts request the same computation.
- **K/V substitution:** Replacing selected Key and Value cache entries with
  entries from another prompt. This differs from activation patching, which
  replaces a hidden-state vector during model processing.
- **FP16:** A 16-bit floating-point number format used here for model weights
  and KV tensors during CUDA runs.
