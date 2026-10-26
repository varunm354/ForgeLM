# BPE Transformer baseline

ForgeLM's BPE Transformer uses the same 4-layer, 128-dimensional decoder-only
architecture as the character baseline, with four causal attention heads,
512-dimensional feed-forward layers, learned absolute positions, pre-norm
residual blocks, and a 128-token context. Its 512-token output vocabulary raises
the parameter count from 824,897 to 939,776.

## Train-only tokenizer

Raw Tiny Shakespeare text was split contiguously before tokenization. BPE merges
were learned from the 90% training portion only, then training and validation
text were encoded separately.

- Vocabulary size: 512
- Learned merges: 256
- Fingerprint:
  `8967b56ee8ddbd52b9af9c638647f738cf34757177458f7077ea35fd175ca4a4`
- Training tokens: 621,873
- Validation tokens: 70,184

Compared with character tokenization, BPE reduced the training sequence by
38.05% and the validation sequence by 37.08%. This tokenizer is distinct from
the Phase 3A full-corpus compression artifact.

## Training configuration

| Setting | Value |
| --- | ---: |
| Device | Apple MPS |
| Parameters | 939,776 |
| Training steps | 5,000 |
| Batch size | 32 |
| Context length | 128 BPE tokens |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.2 |
| Learning rate | 0.0003 |
| AdamW weight decay | 0.01 |
| Evaluation interval | 250 steps |
| Sampled evaluation batches | 20 |

## Selected training milestones

These are sampled token-level training metrics.

| Step | Train loss / perplexity | Validation loss / perplexity |
| ---: | ---: | ---: |
| 0 | 6.3743 / 586.56 | 6.3740 / 586.42 |
| 1,000 | 3.0190 / 20.47 | 3.1418 / 23.14 |
| 2,000 | 2.7309 / 15.35 | 2.9421 / 18.96 |
| 3,000 | 2.5135 / 12.35 | 2.7523 / 15.68 |
| 4,000 | 2.3520 / 10.51 | 2.6445 / 14.08 |
| 5,000 | 2.2860 / 9.84 | 2.5779 / 13.17 |

The best checkpoint was saved at step 5,000.

## Generated sample

```text

I should a had-beat him the long emparried of him,
'Tis the flatter place, and me to his beguiltains,
From our fruit and he make the friends.
Your free blood, weeping every of the plays,
He since of his touch and in our father,
Like in the princountry place of their danger
To be her cold a soulve us for the men;
No pray! it, and that he is the hearts to the sland.

SLON:
Look, that he is a warm at tongue of
Bo formers' where is a tappear, pass of France matter-tongue!
Prance is not contentine offied as as gland,
But this part even to myself, I have a sent mean,
My spack, and as heavens
No wair a crown my cle,
More his son of my best live be brand.

AUTOLYCUS:
As will I fear me who sleeps all the day:
Here you crop in the grown, nor and sweet them.
And, sir, words my
```

The BPE sample contains longer word-like units and more sustained
sentence-shaped structure than the character baseline sample. Generated text
remains semantically inconsistent, and a single sample is qualitative evidence
rather than an evaluation metric.

## Exact full-corpus comparison

Both step-5,000 checkpoints were evaluated on the same raw validation text.
Every target token was scored exactly once using unreduced cross-entropy, and
its exact represented UTF-8 byte length was accumulated independently.

| Measurement | Character Transformer | BPE Transformer |
| --- | ---: | ---: |
| Validation tokens | 111,540 | 70,184 |
| Evaluated target tokens | 111,539 | 70,183 |
| Evaluated target bytes | 111,539 | 111,539 |
| Mean token loss | 1.733565474 | 2.569239000 |
| Token perplexity | 5.660801410 | 13.055885137 |
| Total NLL | 193,360.159408863 | 180,316.900768768 |
| Bits per byte | 2.501006312 | 2.332299003 |

The BPE model improves bits per byte by **0.168707309**, or **6.7456%**, on
this benchmark. Token-level perplexities cannot be compared directly because a
BPE token represents more source bytes on average than a character token.

## Evaluation limitations

Evaluation partitions each token stream into non-overlapping 128-token windows.
Context resets at every window boundary, matching ForgeLM's existing
deterministic benchmark but giving BPE windows broader byte coverage. The first
token is context rather than a scored target. Here both initial validation
tokens represent one byte, so both models score exactly 111,539 bytes; other
corpora can produce a small byte-count difference.

This is an educational 512-token tokenizer and small-scale pretraining run.
Production tokenizers typically use much larger and more representative
corpora, optimized segmentation, normalization, special-token policies, and
substantially larger vocabularies and models.
