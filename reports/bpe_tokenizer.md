# Byte-level BPE tokenizer

Phase 3A adds a deterministic byte-level byte-pair encoding tokenizer built
from scratch. It is evaluated here as a tokenizer only; the ForgeLM Transformer
has not yet been retrained with BPE tokens.

## Why move beyond characters?

Character tokenization is transparent and lossless, but it produces long
sequences and forces a model to reconstruct every word from individual
characters. Longer token sequences increase attention cost and reduce the
amount of text covered by a fixed context window. A character vocabulary also
needs an explicit policy for characters absent from its training corpus.

Byte-level BPE starts with all 256 byte values. UTF-8 represents every valid
input string with those bytes, so any text can be encoded without an unknown
token. Training then repeatedly replaces the most frequent adjacent token pair
with a new sequential token ID. Common byte sequences become single tokens,
shortening the sequence while retaining byte-level fallback.

## Deterministic training

ForgeLM pre-tokenizes with the standard-library regular expression
`\s+|\S+`. Concatenating its whitespace and non-whitespace chunks exactly
reconstructs the source. Merges are learned and applied within chunks only;
they never cross a pre-tokenization boundary.

Duplicate chunks are counted once and assigned a frequency. Pair counts are
weighted by that frequency, and each selected merge is applied to every unique
chunk representation. This avoids rescanning every occurrence of repeated
words separately. Pair-frequency ties are resolved by the lexicographic order
of the pair's integer token IDs, making training reproducible.

During encoding, adjacent learned pairs are selected according to their
original merge rank. Lower-rank merges take priority, and encoding uses the
same pre-tokenization boundaries as training. Token IDs 0–255 always represent
their corresponding bytes; learned IDs begin at 256.

## Tiny Shakespeare results

The training script used the 1,115,394-character Tiny Shakespeare corpus and a
requested vocabulary size of 512.

| Measurement | Result |
| --- | ---: |
| Corpus bytes | 1,115,394 |
| Character tokens | 1,115,394 |
| Achieved vocabulary size | 512 |
| Learned merges | 256 |
| BPE tokens | 691,315 |
| Token-count reduction | 38.02% |
| Bytes per BPE token | 1.6134 |
| Training runtime | 20.16 seconds |
| Full-corpus round trip | Passed |

The generated tokenizer is saved under
`artifacts/tokenizers/tiny_shakespeare_bpe.json`, which is intentionally
ignored by Git.

## Limitations

- The whitespace/non-whitespace split is intentionally simple and does not
  provide language-aware or punctuation-aware pre-tokenization.
- Training rescans unique chunk representations for each merge. Frequency
  compression keeps this practical for Tiny Shakespeare, but specialized data
  structures would scale better to large corpora and vocabularies.
- The tokenizer performs no Unicode normalization and defines no special,
  padding, beginning-of-sequence, or end-of-sequence tokens.
- A 512-token vocabulary trained on Tiny Shakespeare is a pipeline experiment,
  not a broadly representative production vocabulary.

Production tokenizers commonly add highly optimized trainers and encoders,
carefully designed normalization and pre-tokenization, special-token policies,
streaming support, and extensive compatibility guarantees. This custom
implementation prioritizes explicit BPE mechanics, deterministic behavior, and
testability for education.
