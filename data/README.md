# Data

Tiny Shakespeare is the initial corpus for validating ForgeLM's character-level
tokenization and language-model data pipeline. Its small size makes iteration
quick, but it is not intended to represent the scope or quality of the final
training data. Later ForgeLM experiments will use a more substantial dataset.

The source text comes from the Tiny Shakespeare dataset in Andrej Karpathy's
[`char-rnn` repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).
Download it with:

```bash
uv run python scripts/download_tiny_shakespeare.py
```

Source files are stored in `data/raw/`, and transformed datasets belong in
`data/processed/`. Raw and processed data are local artifacts and are not
committed to the repository.
