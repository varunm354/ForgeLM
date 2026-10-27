# GPT 50M production architecture

Phase 4B adds a modern decoder-only GPT architecture intended for later cloud
pretraining on the Phase 4A FineWeb-Edu shards. This phase implements and
validates the architecture only. The model has not been pretrained.

## Educational and production model paths

ForgeLM keeps both model families because they serve different purposes.
`TransformerLanguageModel` and the components in `attention.py` explicitly
show separate query, key, and value projections, causal masking, attention
softmax, and weighted value aggregation. That implementation remains the
readable proof of the underlying mathematics and continues to support the Tiny
Shakespeare experiments unchanged.

`GPTLanguageModel` is a separate production path. It uses fused projections,
PyTorch scaled dot-product attention, rotary positions, RMSNorm, SwiGLU,
bias-free linear layers, and tied token/output weights. These choices reduce
unnecessary operations and make it possible for PyTorch to select optimized
attention kernels on suitable cloud accelerators.

## Default architecture

The checked-in configuration uses a 16,384-token vocabulary, a 1,024-token
maximum sequence, width 512, 12 Transformer blocks, 8 attention heads, a
64-feature head dimension, and a 1,536-feature SwiGLU hidden dimension.
Dropout is zero for the planned pretraining baseline. Linear layers are
bias-free, RoPE uses base 10,000, and token embedding weights are shared with
the bias-free output projection.

The validated configuration rejects nonpositive dimensions, inconsistent head
geometry, odd rotary dimensions, invalid dropout, vocabularies beyond the
65,536-ID `uint16` pipeline limit, and context lengths beyond the precomputed
RoPE capacity.

## RMSNorm

Each block uses pre-normalization with RMSNorm:

`x * rsqrt(mean(x²) + epsilon) * learned_scale`

Unlike LayerNorm, RMSNorm does not subtract the feature mean and has no learned
shift. Its reduction is calculated in float32 for numerical stability, then
returned in the input dtype. A learned scale remains for every embedding
feature.

Pre-normalization places normalization before each residual branch. The block
therefore computes attention from normalized input, adds its residual, computes
SwiGLU from a second normalized state, and adds the second residual. This
layout generally has more stable gradient flow than post-normalization for
deeper decoder stacks.

## Rotary positional embeddings

There is no learned positional embedding table. RoPE rotates pairs of query
and key features using position-specific cosine and sine values. With head
dimension 64, each head contains 32 rotated feature pairs. The cosine and sine
tables are calculated once through the configured 1,024-position capacity,
stored as non-parameter buffers, and cast to the active device and dtype when
used.

Pairwise rotation preserves vector norms and injects relative position into
query-key dot products. It applies only to queries and keys; values are
unchanged.

## Optimized causal attention

For hidden input shaped `(B, T, 512)`, one fused linear layer produces
`(B, T, 1536)`. It is viewed as `(B, T, 3, 8, 64)` and permuted into query,
key, and value tensors shaped `(B, 8, T, 64)`. RoPE transforms query and key,
then `scaled_dot_product_attention` runs with `is_causal=True`.

The attention dropout argument is the configured probability during training
and exactly zero during evaluation. PyTorch can dispatch this operation to an
optimized backend appropriate for the active device and tensor type. The
result returns through `(B, T, 512)` and a residual output projection.

A fused QKV projection performs one larger matrix multiplication instead of
three smaller launches. It also makes the projection output contiguous for the
following reshape, improving accelerator utilization while preserving the same
mathematics. CPU tests compare the module against an explicit masked-softmax
reference with dropout disabled.

## SwiGLU feed-forward network

The feed-forward path has separate gate and value projections:

`(B, T, 512) -> gate/value (B, T, 1536)`

The gate receives SiLU activation and is multiplied elementwise with the value
branch. An output projection maps `(B, T, 1536)` back to `(B, T, 512)`. SwiGLU
adds data-dependent gating while the 1,536 hidden width keeps the target model
near 50 million parameters.

## Complete language model

Token IDs shaped `(B, T)` become embeddings shaped `(B, T, 512)`, pass through
12 blocks and a final RMSNorm, and produce logits shaped `(B, T, 16384)`.
Optional targets with shape `(B, T)` use flattened next-token cross-entropy.
Sequences longer than 1,024 are rejected during forward evaluation.

The language-model projection is bias-free and points to the exact same
parameter storage as the token embedding. Weight tying avoids a second
8,388,608-parameter vocabulary matrix and encourages compatible input and
output token representations.

Generation is autoregressive and crops context to the configured block size.
It is sufficient for architecture smoke tests, but intentionally has no
key-value cache yet.

## Initialization and exact parameter count

Embeddings and linear weights begin from a normal distribution with standard
deviation 0.02. Attention and SwiGLU residual output projections use
`0.02 / sqrt(2 * 12)`. Two residual branches are accumulated per layer, so
this GPT-style depth scaling controls their initial variance.

The tied token embedding contributes 8,388,608 unique parameters. Each block
contains 1,048,576 attention projection parameters, 2,359,296 SwiGLU
parameters, and 1,024 RMSNorm scale parameters, totaling 3,408,896 per block.
Twelve blocks contribute 40,906,752. The final RMSNorm contributes 512.
RoPE buffers contain no trainable parameters, and the tied output projection
adds no unique parameters.

The implementation-calculated total is therefore **49,295,872 trainable
parameters**. Shared storage is counted once.

Parameter-only storage is:

- float32: 197,183,488 bytes, or 188.05 MiB
- float16: 98,591,744 bytes, or 94.02 MiB
- bfloat16: 98,591,744 bytes, or 94.02 MiB

These figures exclude gradients, optimizer states, activations, attention
workspace, allocator overhead, checkpoints, and distributed-training copies.
Actual training memory will be substantially higher.

## Remaining work before cloud pretraining

Phase 4B does not provide the cloud training loop. Remaining work includes
mixed-precision policy, optimizer and learning-rate schedule, gradient
accumulation and clipping, distributed data parallelism, deterministic
memory-mapped shard iteration, resumable model/optimizer checkpoints,
validation and metric reporting, accelerator throughput profiling, and
failure-safe cloud artifact storage.

Phase 4D now pins the FineWeb-Edu revision, but the production tokenizer
artifact and fingerprint remain unbuilt and the planned 500-million-token
Phase 4A corpus remains unprocessed. Later inference work can add a KV cache
and serving-specific generation controls without changing this architecture's
training semantics.
