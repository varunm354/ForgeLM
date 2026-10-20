# Bigram baseline

The first ForgeLM baseline is a character-level bigram language model trained
on Tiny Shakespeare. It predicts each next character using only the immediately
preceding character.

## Experiment

| Setting | Value |
| --- | ---: |
| Dataset | Tiny Shakespeare |
| Character count | 1,115,394 |
| Vocabulary size | 65 |
| Training tokens | 1,003,854 |
| Validation tokens | 111,540 |
| Model parameters | 4,225 |
| Training steps | 3,000 |
| Batch size | 32 |
| Block size | 8 |
| Learning rate | 0.01 |
| Device | CPU |
| Runtime | 9.51 seconds |

## Results

| Measurement | Train | Validation |
| --- | ---: | ---: |
| Initial loss | 4.7455 | 4.7017 |
| Initial perplexity | 115.06 | 110.13 |
| Final loss | 2.4546 | 2.4597 |
| Final perplexity | 11.64 | 11.70 |

The sharp early loss reduction followed by a stable plateau shows that the
model converged to the useful statistics available to a bigram model. The
nearly equal final training and validation metrics indicate similar behavior on
held-out text rather than substantial overfitting. Its strict one-character
context remains the central limitation: generated text can reproduce local
character transitions and word-like fragments, but it cannot model longer
spelling, syntax, or semantic relationships.

## Final generated sample

```text

TI wicecaveotave heand kert his he be; larorirkengherbe utobungoeved?
GUG ack,
COWhapens har the, pers paly lom, phe mou wid pt haAw oRKETI oupielego le rdXIUENTY:
Fis ran s, o ig rd.
che knd etndof
```
