import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"

st.set_page_config(page_title="Signal — earnings-call risk", page_icon="◉", layout="wide")

CSS = """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
h1, h2, h3 {letter-spacing: -0.02em;}
.hero {background: linear-gradient(135deg, #0f172a 0%, #1e293b 60%, #0b3b3c 100%);
       color: #f8fafc; padding: 1.5rem 1.8rem; border-radius: 14px; margin-bottom: 1.2rem;}
.hero h1 {color: #f8fafc; margin: 0 0 .25rem 0; font-size: 2.1rem;}
.hero p {color: #cbd5e1; margin: 0; font-size: .95rem;}
.chip {display:inline-block; background:#0f766e; color:#ecfeff; padding:.18rem .55rem;
       border-radius:999px; font-size:.72rem; font-weight:600; letter-spacing:.04em; margin-right:.35rem;}
.card {border:1px solid #e2e8f0; border-radius:12px; padding:1rem 1.1rem; background:#ffffff;}
.big {font-size:2.2rem; font-weight:700; line-height:1; margin:.2rem 0;}
.muted {color:#64748b; font-size:.82rem;}
.pos {color:#b91c1c; font-weight:600;}
.neg {color:#0f766e; font-weight:600;}
.driver {padding:.45rem .6rem; border-radius:8px; margin-bottom:.3rem; background:#f8fafc;
         border-left:4px solid #cbd5e1; font-size:.86rem;}
.driver.up {border-left-color:#b91c1c;}
.driver.down {border-left-color:#0f766e;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load():
    calls = pd.read_parquet(A / "demo_calls.parquet")
    drivers = pd.read_parquet(A / "demo_drivers.parquet")
    prosody = pd.read_parquet(A / "demo_prosody.parquet")
    metrics = pd.read_csv(A / "metrics.csv")
    fi = pd.read_csv(A / "feature_importance.csv")
    return calls, drivers, prosody, metrics, fi


calls, drivers, prosody, metrics, fi = load()
calls["date"] = pd.to_datetime(calls["date"])

LABELS = {
    "pre_vol": "Pre-call volatility",
    "n_segments": "Call length (segments)",
    "audio_len": "Audio length (seconds)",
    "hedge_rate": "Hedging language",
    "neg_rate": "Negative language",
    "fwd_rate": "Forward-looking language",
    "modal_rate": "Modal verbs",
    "ttr": "Lexical diversity",
    "n_numbers": "Numeric density",
    "sent_negative": "FinBERT negative sentiment",
    "sent_positive": "FinBERT positive sentiment",
    "sent_neutral": "FinBERT neutral sentiment",
    "acc_Standard deviation of period_std": "Period variability",
    "acc_Mean period_std": "Period variability",
    "acc_Degree of voice breaks_max": "Voice breaks",
    "acc_Jitter local absolute_max": "Absolute jitter",
    "dyn_Mean HNR_diffstd": "HNR change variability",
    "dyn_Fraction of unvoiced_diffstd": "Unvoiced change variability",
}

MODEL_NAMES = {
    "late_pre_text_ac": "Late fusion (pre-vol + text + voice)",
    "late_pre_text": "Late fusion (pre-vol + text)",
    "pre_vol": "Pre-call volatility only",
    "late_all": "Late fusion (all four models)",
    "fusion": "Single LightGBM (all features)",
    "fusion_no_emb": "Single LightGBM (no FinBERT)",
    "text_tfidf": "Text only (TF-IDF + lexicon)",
    "text_emb": "Text only (FinBERT embeddings)",
    "acoustic": "Voice only (acoustic descriptors)",
}


def human(feature: str) -> str:
    if feature in LABELS:
        return LABELS[feature]
    if feature.startswith("svd_"):
        return "Text style component"
    if feature.startswith("emb_"):
        return "FinBERT embedding"
    f = feature
    for prefix in ("acc_", "dyn_", "d_"):
        if f.startswith(prefix):
            f = f[len(prefix):]
            break
    f = f.replace("_", " ")
    f = re.sub(r"\bmean\b", "average", f)
    f = re.sub(r"\bstd\b", "variability", f)
    f = re.sub(r"\bmax\b", "peak", f)
    f = re.sub(r"\bmin\b", "low", f)
    f = f.replace("jitter local", "vocal jitter").replace("shimmer local", "vocal shimmer")
    f = f.replace("hnr", "harmonics-to-noise").replace("fraction of unvoiced", "unvoiced fraction")
    return f[:1].upper() + f[1:]


def fmt_value(feature: str, value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if feature in ("pre_vol", "hedge_rate", "neg_rate", "fwd_rate", "modal_rate", "ttr"):
        return f"{value:.3f}"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    return f"{value:.3f}"


st.markdown(
    """
<div class="hero">
  <span class="chip">MULTIMODAL</span><span class="chip">EVALUATED</span><span class="chip">REPRODUCIBLE</span>
  <h1>Signal</h1>
  <p>Will this earnings call move the stock? A text-and-voice nowcast of post-call volatility,
     evaluated on held-out calls from 2017–2018.</p>
</div>
""",
    unsafe_allow_html=True,
)

best = metrics.sort_values("val_auc", ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Selected model ROC-AUC",
    f"{best.auc:.3f}",
    help="Chosen on validation AUC, then scored once on the held-out test set",
)
c2.metric("Text-only ROC-AUC", f"{metrics.loc[metrics.model == 'text_tfidf', 'auc'].iloc[0]:.3f}")
c3.metric("Voice-only ROC-AUC", f"{metrics.loc[metrics.model == 'acoustic', 'auc'].iloc[0]:.3f}")
c4.metric("Held-out calls", f"{len(calls):,}")

tab_explorer, tab_perf, tab_method = st.tabs(["Call explorer", "Model performance", "Method & limitations"])

with tab_explorer:
    tickers = sorted(calls.ticker.unique())
    left, right = st.columns([1, 3])
    with left:
        default = tickers.index("PEP") if "PEP" in tickers else 0
        ticker = st.selectbox("Ticker", tickers, index=default)
        subset = calls[calls.ticker == ticker].sort_values("date")
        options = [f"{r.date.date()}  ·  {r.call_id}" for r in subset.itertuples()]
        choice = st.selectbox("Call", options, index=len(options) - 1)
        call_id = choice.split("·")[-1].strip()
        row = calls[calls.call_id == call_id].iloc[0]
        risk_display = "<1" if row.risk < 0.01 else f"{row.risk * 100:.0f}"

        st.markdown(
            f"<div class='card'><div class='muted'>RISK SCORE</div>"
            f"<div class='big'>{risk_display}<span style='font-size:1rem'> / 100</span></div>"
            f"<div class='muted'>Percentile {row.risk_pct * 100:.0f} of held-out calls</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div class='card'><div class='muted'>ACTUAL OUTCOME</div>"
            f"<div style='font-size:1.15rem;font-weight:600'>{row.outcome}</div>"
            f"<div class='muted'>Post-call vol {row.post_vol:.1%} · pre-call {row.pre_vol:.1%}</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div class='card'><div class='muted'>BASELINES</div>"
            f"<div style='font-size:1.02rem'>Pre-call volatility: <b>{row.pred_pre_vol:.1%}</b></div>"
            f"<div class='muted'>Text only {row.pred_text_tfidf:.1%} · Voice only {row.pred_acoustic:.1%}</div>"
            f"<div class='muted'>Predicted probability of a high-volatility call</div></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("What drives the score")
        d = drivers[drivers.call_id == call_id].copy()
        d["abs"] = d.contribution.abs()
        d = d.sort_values("abs", ascending=False).head(6)
        for r in d.itertuples():
            direction = "up" if r.contribution > 0 else "down"
            sign = "+" if r.contribution > 0 else ""
            cls = "pos" if r.contribution > 0 else "neg"
            st.markdown(
                f"<div class='driver {direction}'><b>{human(r.feature)}</b> = {fmt_value(r.feature, r.value)}"
                f"<span style='float:right' class='{cls}'>{sign}{r.contribution:.2f}</span></div>",
                unsafe_allow_html=True,
            )
        st.caption("SHAP contributions from the fusion model. Positive pushes toward high volatility.")

        st.subheader("Delivery across the call")
        p = prosody[prosody.call_id == call_id]
        if len(p):
            fig = go.Figure()
            for metric, color, axis in [
                ("Mean pitch", "#0f766e", "y1"),
                ("Mean intensity", "#b45309", "y2"),
            ]:
                s = p[p.metric == metric].sort_values("segment")
                if len(s):
                    fig.add_trace(
                        go.Scatter(x=s.segment, y=s.value, name=metric, mode="lines",
                                   line={"color": color, "width": 2}, yaxis=axis)
                    )
            fig.update_layout(
                height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10},
                yaxis={"title": "Pitch (Hz)", "showgrid": False},
                yaxis2={"title": "Intensity (dB)", "overlaying": "y", "side": "right", "showgrid": False},
                legend={"orientation": "h", "y": 1.12},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No segment-level prosody available for this call.")

        with st.expander("Transcript excerpt", expanded=False):
            st.write(row.snippet)

with tab_perf:
    st.subheader("Held-out performance")
    show = metrics.sort_values("val_auc", ascending=False).copy()
    show["model"] = show.model.map(lambda m: MODEL_NAMES.get(m, m))
    show["Val AUC"] = show.val_auc.round(3)
    show["AUC (95% CI)"] = show.apply(lambda r: f"{r.auc:.3f} ({r.auc_lo:.3f}–{r.auc_hi:.3f})", axis=1)
    show["PR-AUC"] = show.ap.round(3)
    show["Brier"] = show.brier.round(3)
    show["Log loss"] = show.logloss.round(3)
    st.dataframe(
        show[["model", "Val AUC", "AUC (95% CI)", "PR-AUC", "Brier", "Log loss"]].rename(
            columns={"model": "Model"}
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        "Sorted by validation AUC; the first row is the shipped model. "
        "Time split: train 2015–2016, validation 2017 H1, test 2017 H2–2018 H1. "
        "Intervals from 1,000 bootstrap resamples."
    )

    a, b = st.columns(2)
    with a:
        st.subheader("ROC curves")
        fig = go.Figure()
        for model, color, label in [
            ("pred_late_pre_text_ac", "#0f766e", "Late fusion (text + voice + pre-vol)"),
            ("pred_pre_vol", "#64748b", "Pre-call volatility"),
            ("pred_text_tfidf", "#b45309", "Text only"),
            ("pred_acoustic", "#7c3aed", "Voice only"),
        ]:
            if model not in calls:
                continue
            fpr, tpr, _ = roc_curve(calls.y, calls[model])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=label, line={"color": color, "width": 2}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={"color": "#cbd5e1", "dash": "dash"}, showlegend=False))
        fig.update_layout(height=360, xaxis_title="False positive rate", yaxis_title="True positive rate",
                          margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.subheader("Calibration")
        fig = go.Figure()
        for model, color in [("pred_late_pre_text_ac", "#0f766e"), ("pred_pre_vol", "#64748b")]:
            if model not in calls:
                continue
            frac, mean = calibration_curve(calls.y, calls[model], n_bins=8, strategy="quantile")
            fig.add_trace(go.Scatter(x=mean, y=frac, mode="lines+markers", name=model.replace("pred_", ""),
                                     line={"color": color, "width": 2}))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line={"color": "#cbd5e1", "dash": "dash"}, showlegend=False))
        fig.update_layout(height=360, xaxis_title="Predicted risk", yaxis_title="Observed high-vol rate",
                          margin={"l": 10, "r": 10, "t": 10, "b": 10}, legend={"orientation": "h", "y": -0.25})
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Which modalities earn their place?")
    abl = metrics.sort_values("val_auc")
    fig = go.Figure(go.Bar(
        x=abl.auc, y=abl.model.map(lambda m: MODEL_NAMES.get(m, m)), orientation="h",
        marker_color=["#0f766e" if m == best.model else "#cbd5e1" for m in abl.model],
        text=abl.auc.round(3), textposition="outside",
    ))
    fig.update_layout(height=320, xaxis={"range": [0.45, 0.85], "title": "Test ROC-AUC"},
                      margin={"l": 10, "r": 30, "t": 10, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Pre-call volatility does most of the work. Text has standalone signal but adds about 0.001 AUC "
        "over the baseline; adding voice lowers test AUC. Language carries what little incremental signal "
        "there is, not hand-crafted prosody."
    )

    st.subheader("Top features by gain")
    top = fi.head(15).sort_values("gain")
    st.dataframe(
        top.assign(feature=top.feature.map(human))[["feature", "gain"]].rename(columns={"feature": "Feature", "gain": "Gain"}),
        hide_index=True, use_container_width=True,
    )

with tab_method:
    st.subheader("The question")
    st.markdown(
        "Earnings calls are the densest unstructured signal a public company emits. This project asks a narrow, "
        "testable version of a large question: **does the way management speaks — and the language they use — "
        "forecast how volatile the stock will be after the call?**"
    )
    st.subheader("Data")
    st.markdown(
        "- **MAEC** (CIKM 2020): 3,443 earnings calls from S&P 1500 companies, each with a transcript and 29 "
        "acoustic descriptors per segment. CC BY-SA 4.0; not redistributed here.\n"
        "- **Prices**: daily closes from yfinance; SPY as the market benchmark.\n"
        "- **Labels**: top tercile of post-call realised volatility, measured over the five trading days after "
        "the call, market-adjusted against SPY. The threshold is learned on the training period only. A "
        "post/pre ratio target was tested and rejected because it removes the predictable volatility level."
    )
    st.subheader("Protocol")
    st.markdown(
        "- Time split: train 2015–2016, validation 2017 H1, test 2017 H2–2018 H1. No random splits.\n"
        "- Company-disjoint cross-validation as a robustness check, because the same firm appears repeatedly.\n"
        "- Text representations (TF-IDF, SVD) are fit on training data only.\n"
        "- Probabilities are isotonic-calibrated on the validation period.\n"
        "- Confidence intervals from 1,000 bootstrap resamples of the test set."
    )
    st.subheader("What the ablation says")
    st.markdown(
        "Pre-call volatility is a strong baseline (0.776 test AUC), because volatility clusters. Text alone "
        "reaches 0.674, but adds only about 0.001 AUC on top of the baseline. Voice alone is 0.543 — close to "
        "chance — and adding it lowers test AUC from 0.780 to 0.776. That is a real finding, not a bug: "
        "hand-crafted prosody features do not generalise across speakers and recording conditions, and the "
        "label is dominated by market-wide moves. A production system would need speaker-normalised audio "
        "embeddings and a cleaner surprise label."
    )
    st.subheader("Limitations")
    st.markdown(
        "- The label is noisy: five-day realised volatility is affected by macro news unrelated to the call.\n"
        "- MAEC's transcripts have no speaker labels, so Q&A-specific acoustics are approximated, not isolated.\n"
        "- The corpus ends in 2018; modern calls, accents and audio quality may behave differently.\n"
        "- This is a research prototype, not investment advice."
    )
    st.subheader("Reproduce")
    st.code(
        "git clone <repo>\n"
        "git clone --depth 1 https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction.git data/maec\n"
        "python build_features.py && python build_dynamics.py && python build_deltas.py\n"
        "python download_prices.py && python build_labels.py\n"
        "python embed_text.py   # optional FinBERT features\n"
        "python model.py && python build_demo.py\n"
        "streamlit run app/dashboard.py",
        language="bash",
    )
