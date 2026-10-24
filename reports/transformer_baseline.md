# Transformer baseline

ForgeLM's first decoder-only Transformer baseline is a 4-layer,
824,897-parameter character language model. It uses learned token and absolute
positional embeddings, four explicit causal attention heads per pre-norm block,
a 4× position-wise feed-forward expansion, residual connections, and a final
normalization and language-model head.

## Dataset and configuration

The model was trained on Tiny Shakespeare, containing 1,115,394 characters and
a 65-character vocabulary. The contiguous 90/10 split contains 1,003,854
training tokens and 111,540 validation tokens.

| Setting | Value |
| --- | ---: |
| Device | Apple MPS |
| Parameters | 824,897 |
| Training steps | 5,000 |
| Batch size | 32 |
| Context length | 128 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.2 |
| Learning rate | 0.0003 |
| AdamW weight decay | 0.01 |
| Evaluation interval | 250 steps |
| Sampled evaluation batches | 20 |

## Selected training milestones

These values are sampled training-time estimates, not deterministic
full-corpus metrics.

| Step | Train loss / perplexity | Validation loss / perplexity |
| ---: | ---: | ---: |
| 0 | 4.3271 / 75.72 | 4.3211 / 75.27 |
| 1,000 | 2.1570 / 8.64 | 2.1928 / 8.96 |
| 2,000 | 1.8596 / 6.42 | 1.9586 / 7.09 |
| 3,000 | 1.7014 / 5.48 | 1.8611 / 6.43 |
| 4,000 | 1.6192 / 5.05 | 1.7975 / 6.03 |
| 5,000 | 1.5454 / 4.69 | 1.7348 / 5.67 |

The best checkpoint was saved at step 5,000.

## Comparison with the bigram baseline

On the sampled final evaluations, the Transformer reduced validation loss from
2.4597 to 1.7348, a **29.47% reduction**, and validation perplexity from 11.70
to 5.67, a **51.54% reduction**. These figures compare the completed sampled
training runs. They must not be mixed with the deterministic full-corpus
metrics produced by `scripts/compare_baselines.py`.

### Deterministic full-corpus benchmark

The separate benchmark evaluated each of the 111,539 validation targets exactly
once using the best step-3,000 Bigram and step-5,000 Transformer checkpoints.

| Model | Validation loss | Validation perplexity |
| --- | ---: | ---: |
| Bigram | 2.486503 | 12.019169 |
| Transformer | 1.733565 | 5.660801 |
| Reduction | 30.28% | 52.90% |

These deterministic values use token-weighted cross-entropy over contiguous,
non-overlapping windows and are intentionally reported separately from the
sampled metrics above.

Unlike the one-character-context bigram, the Transformer learned longer local
structure: speaker-like labels, line breaks, punctuation patterns, word
fragments, and sentence-shaped sequences. The generated text is still not
consistently grammatical or coherent, but its longer-range organization is
substantially stronger than the bigram sample.

## Generated sample

```text

God to him me the gue? that not side, a will.

MENENIUS:
Aladam, and but you should let's like my somer!

POLIXENES:
Well getrantleman being my on me ars--
Not lord, I no into be his rahed of our purford,
One my did and thou galle of you?

ELEONTES:
Good your is say, Edward decitings
I will that gens say your lady undang breath forling comming
Perfore of our than yet of thee, enten yet.

ELEO:
Thou conswer,
And The time-parturn way, welcome mercist did reash;
Before is lifefting and cursent us o
```

## Limitations

- Character-level tokenization is simple and transparent, but sequences are
  longer and lexical structure must be learned character by character.
- Tiny Shakespeare is narrow, stylistically repetitive, and far smaller than a
  modern language-model corpus.
- An 824,897-parameter model has limited capacity.
- The 128-token context cannot represent long scenes or broader document
  structure.

This run is an educational, small-scale pretraining baseline. It validates the
ForgeLM model and training pipeline; it is not the project's final scale target.
