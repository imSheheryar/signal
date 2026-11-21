# Model card — Signal

## Model details

- **Model:** late-fusion logistic stack of three calibrated risk models: pre-call volatility, TF-IDF transcript text, and acoustic prosody.
- **Version:** 0.1.0.
- **Task:** predict whether an earnings call will be followed by top-tercile market-adjusted volatility over the next five trading days.
- **Output:** probability in `[0.01, 0.99]`, calibrated on a held-out validation period.
- **Selection:** highest validation ROC-AUC among seven candidates; the test set was scored once after selection.
- **Artifacts:** `artifacts/models.joblib`, `artifacts/metrics.csv`, `artifacts/model_meta.json`.

## Intended use

Research and portfolio demonstration: ranking historical earnings calls for analyst review, and showing how much signal comes from language, voice and market context separately. Suitable for evaluating multimodal feature ideas on MAEC.

## Out-of-scope use

- Live trading or investment advice.
- Scoring calls after 2018 without revalidation.
- Judging individual executives, speakers or firms.
- Any decision with legal, employment or financial consequences for a person.

## Training data

- 2,401 labelled calls from 834 US-listed tickers, 2015-02-25 to 2018-06-05.
- MAEC transcripts and 29 per-segment acoustic descriptors; yfinance daily closes and SPY.
- Split by time: 1,497 train (2015–2016), 455 validation (2017 H1), 449 test (2017 H2–2018 H1).
- No random split. No firm appears in both train and test by construction of the temporal split; a company-disjoint 5-fold CV is reported as a second check.

## Features

- **Pre-call volatility:** 60 trading days of market-adjusted returns ending the day before the call.
- **Text:** TF-IDF 30k features (1–2 grams) reduced to 100 SVD components; hedging, negation, forward-looking and modal rates; FinBERT sentiment probabilities; optional 768-d FinBERT mean embeddings.
- **Voice:** 29 descriptors aggregated per call, within-call slopes and third-to-third deltas, and prior-call changes.
- **Metadata:** segment count, audio length.

## Metrics

Test set, 449 calls, base rate 38.5%:

| Model | ROC-AUC | 95% CI | PR-AUC | Brier |
|---|---|---|---|---|
| Late fusion (shipped) | 0.776 | 0.734–0.823 | 0.677 | 0.198 |
| Late fusion, no voice | 0.780 | 0.736–0.824 | 0.681 | 0.194 |
| Pre-call volatility only | 0.776 | 0.733–0.822 | 0.663 | 0.198 |
| Text only (TF-IDF) | 0.674 | 0.637–0.737 | 0.532 | 0.228 |
| Voice only | 0.543 | 0.521–0.630 | 0.408 | 0.254 |

- **Top-20% review:** 71.1% precision, 37.0% recall, 1.85× lift for the shipped model; 75.6% precision, 39.3% recall, 1.96× lift without voice.
- **Company-disjoint CV:** 0.738 ± 0.024 AUC on the single LightGBM fusion.
- **Calibration:** isotonic on validation; probabilities clipped to `[0.01, 0.99]`.

## Limitations

- Five-day realised volatility is a noisy label; macro and sector shocks dominate.
- The corpus ends in 2018 and covers US-listed firms only.
- MAEC transcripts lack speaker labels, so Q&A cannot be isolated cleanly.
- Acoustic features are hand-crafted Praat descriptors; self-supervised audio embeddings were not tested.
- FinBERT mean embeddings underperform TF-IDF on this corpus size.
- The API serves the evaluated test set; it is not a live inference service.

## Ethical and reporting notes

The model is not designed to infer the emotional state, honesty or competence of individual speakers. Acoustic and linguistic differences vary by accent, gender, recording setup and dialect; aggregate call-level features may encode those differences. Any production use would require subgroup evaluation and a clear appeal path for affected people.

## Maintenance

Re-run `model.py` after any corpus or feature change; revalidate the calibration period and the top-tercile threshold on new training data. A drift check should compare score distributions and top-decile composition quarter over quarter.
