# Exploratory scripts

These are the scripts that shaped the design, kept for transparency rather than
as part of the reproducible pipeline. Run them from the repository root so the
relative `artifacts/` paths resolve:

```bash
python exploratory/target_test.py
```

- `audit_maec.py` — corpus inventory and date/ticker checks.
- `prelim.py`, `prelim2.py` — first-pass acoustic and lexical associations.
- `target_test.py` — ratio versus volatility-level targets.
- `residual_test.py` — residualised ratio after controlling pre-call volatility.
- `tfidf_test.py` — early TF-IDF signal check.
- `posthigh_test.py` — the post-call volatility level target that shipped.
- `train.py` — an early model sketch, superseded by `model.py`.
