const MODEL_NAMES = {
  late_pre_text_ac: "Late fusion (pre-vol + text + voice)",
  late_pre_text: "Late fusion (pre-vol + text)",
  pre_vol: "Pre-call volatility only",
  late_all: "Late fusion (all four models)",
  fusion_no_emb: "Single LightGBM (no FinBERT)",
  fusion: "Single LightGBM (all features)",
  text_tfidf: "Text only (TF-IDF + lexicon)",
  text_emb: "Text only (FinBERT embeddings)",
  acoustic: "Voice only (acoustic descriptors)",
};

const FEATURE_NAMES = {
  pre_vol: "Pre-call volatility",
  n_segments: "Call length (segments)",
  audio_len: "Audio length (seconds)",
  hedge_rate: "Hedging language",
  neg_rate: "Negative language",
  fwd_rate: "Forward-looking language",
  modal_rate: "Modal verbs",
  ttr: "Lexical diversity",
  n_numbers: "Numeric density",
  sent_negative: "FinBERT negative sentiment",
  sent_positive: "FinBERT positive sentiment",
  sent_neutral: "FinBERT neutral sentiment",
};

const FEATURE_OVERRIDES = {
  "acc_Standard deviation of period_std": "Period variability",
  "acc_Mean period_mean": "Mean period",
  "acc_Mean period_std": "Period variability",
  "acc_Degree of voice breaks_max": "Voice breaks",
  "acc_Degree of voice breaks_mean": "Voice breaks",
  "acc_Jitter local absolute_max": "Absolute jitter",
  "acc_Jitter local absolute_std": "Absolute jitter variability",
  "acc_Mean HNR_std": "Harmonics-to-noise variability",
  "acc_Mean autocorrelation_std": "Autocorrelation variability",
  "acc_Number of periods_std": "Period count variability",
  "acc_Number of pulses_std": "Pulse count variability",
  "dyn_Mean HNR_diffstd": "HNR change variability",
  "dyn_Fraction of unvoiced_diffstd": "Unvoiced change variability",
};

const state = { data: null, callsByTicker: new Map(), current: null, performanceDrawn: false };

const $ = (id) => document.getElementById(id);

function humanFeature(feature) {
  if (FEATURE_NAMES[feature]) return FEATURE_NAMES[feature];
  if (FEATURE_OVERRIDES[feature]) return FEATURE_OVERRIDES[feature];
  if (feature.startsWith("svd_")) return "Text style component";
  if (feature.startsWith("emb_")) return "FinBERT embedding";
  let f = feature;
  for (const prefix of ["acc_", "dyn_", "d_"]) {
    if (f.startsWith(prefix)) { f = f.slice(prefix.length); break; }
  }
  f = f.replaceAll("_", " ");
  f = f.replace(/\bmean\b/g, "average").replace(/\bstd\b/g, "variability")
       .replace(/\bmax\b/g, "peak").replace(/\bmin\b/g, "low")
       .replace("jitter local", "vocal jitter").replace("shimmer local", "vocal shimmer")
       .replace("hnr", "harmonics-to-noise").replace("fraction of unvoiced", "unvoiced fraction");
  return f.charAt(0).toUpperCase() + f.slice(1);
}

function fmtValue(feature, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "n/a";
  if (Math.abs(value) >= 100) return value.toLocaleString(undefined, { maximumFractionDigits: 1 });
  return Number(value).toFixed(3);
}

function pct(value, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function init(data) {
  state.data = data;
  $("statCalls").textContent = data.calls.length.toLocaleString();

  const selected = data.metrics.find((m) => m.model === "late_pre_text_ac");
  const text = data.metrics.find((m) => m.model === "text_tfidf");
  const voice = data.metrics.find((m) => m.model === "acoustic");
  if (selected) $("statAuc").textContent = selected.auc.toFixed(3);
  if (text) $("statText").textContent = text.auc.toFixed(3);
  if (voice) $("statVoice").textContent = voice.auc.toFixed(3);

  for (const call of data.calls) {
    if (!state.callsByTicker.has(call.ticker)) state.callsByTicker.set(call.ticker, []);
    state.callsByTicker.get(call.ticker).push(call);
  }
  for (const list of state.callsByTicker.values()) list.sort((a, b) => a.date.localeCompare(b.date));

  const tickers = [...state.callsByTicker.keys()].sort();
  const tickerSelect = $("tickerSelect");
  tickerSelect.innerHTML = tickers.map((t) => `<option value="${t}">${t}</option>`).join("");
  tickerSelect.value = tickers.includes("PEP") ? "PEP" : tickers[0];
  tickerSelect.addEventListener("change", () => populateCalls(tickerSelect.value));
  $("callSelect").addEventListener("change", () => renderCall($("callSelect").value));

  populateCalls(tickerSelect.value);
  buildMetricsTable();
  bindTabs();
}

function populateCalls(ticker) {
  const list = state.callsByTicker.get(ticker) || [];
  const select = $("callSelect");
  select.innerHTML = list.map((c) => `<option value="${c.call_id}">${c.date} · ${c.call_id}</option>`).join("");
  if (list.length) {
    select.value = list[list.length - 1].call_id;
    renderCall(select.value);
  }
}

function renderCall(callId) {
  const call = state.data.calls.find((c) => c.call_id === callId);
  if (!call) return;
  state.current = call;

  const risk = Math.round(call.risk * 100);
  $("riskScore").textContent = risk < 1 ? "<1" : risk;
  $("riskFill").style.width = `${Math.max(2, call.risk * 100)}%`;
  $("riskPercentile").textContent = `Percentile ${Math.round(call.risk_pct * 100)} of held-out calls`;

  $("outcome").textContent = call.outcome;
  $("vols").textContent = `Post-call vol ${pct(call.post_vol)} · pre-call ${pct(call.pre_vol)}`;
  $("baselinePre").innerHTML = `Pre-call baseline: <strong>${pct(call.pred_pre_vol)}</strong>`;
  $("baselineOther").textContent = `Text-only baseline ${pct(call.pred_text)} · Voice-only baseline ${pct(call.pred_voice)}`;
  $("transcript").textContent = call.snippet;

  const drivers = call.drivers || [];
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 0.0001);
  $("drivers").innerHTML = drivers.map((d) => {
    const positive = d.contribution >= 0;
    const width = Math.max(3, (Math.abs(d.contribution) / maxAbs) * 100);
    const sign = positive ? "+" : "";
    return `<div class="driver ${positive ? "up" : "down"}">
      <div><span class="driver-name">${humanFeature(d.feature)}</span><span class="driver-value">= ${fmtValue(d.feature, d.value)}</span></div>
      <div class="driver-right"><span class="driver-contrib">${sign}${d.contribution.toFixed(2)}</span>
      <div class="driver-bar"><span style="width:${width}%"></span></div></div>
    </div>`;
  }).join("");

  drawProsody(call);
}

function baseLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, sans-serif", color: "#334155", size: 11 },
    margin: { l: 48, r: 48, t: 12, b: 40 },
    hoverlabel: { bgcolor: "#0f172a", font: { color: "#f8fafc" } },
  };
}

function drawProsody(call) {
  if (!window.Plotly) return;
  const pitch = call.prosody["Mean pitch"] || [];
  const intensity = call.prosody["Mean intensity"] || [];
  if (!pitch.length && !intensity.length) {
    $("prosodyChart").innerHTML = '<p class="muted" style="padding:1rem">No segment-level prosody for this call.</p>';
    return;
  }
  const x = pitch.map((_, i) => i);
  const traces = [
    { x, y: pitch, name: "Mean pitch", type: "scatter", mode: "lines", line: { color: "#0f766e", width: 1.8 }, yaxis: "y" },
    { x: intensity.map((_, i) => i), y: intensity, name: "Mean intensity", type: "scatter", mode: "lines", line: { color: "#b45309", width: 1.8 }, yaxis: "y2" },
  ];
  const layout = {
    ...baseLayout(),
    height: 300,
    margin: { l: 48, r: 48, t: 8, b: 36 },
    xaxis: { title: "Segment", showgrid: false, zeroline: false },
    yaxis: { title: "Pitch (Hz)", showgrid: true, gridcolor: "#eef2f7", zeroline: false },
    yaxis2: { title: "Intensity (dB)", overlaying: "y", side: "right", showgrid: false, zeroline: false },
    legend: { orientation: "h", y: 1.12, x: 0, font: { size: 10 } },
  };
  Plotly.react("prosodyChart", traces, layout, { displayModeBar: false, responsive: true });
}

function buildMetricsTable() {
  const body = document.querySelector("#metricsTable tbody");
  body.innerHTML = state.data.metrics.map((m) => {
    const selected = m.model === "late_pre_text_ac";
    return `<tr class="${selected ? "selected" : ""}">
      <td>${MODEL_NAMES[m.model] || m.model}</td>
      <td>${m.val_auc.toFixed(3)}</td>
      <td>${m.auc.toFixed(3)} (${m.auc_lo.toFixed(3)}–${m.auc_hi.toFixed(3)})</td>
      <td>${m.ap.toFixed(3)}</td>
      <td>${m.brier.toFixed(3)}</td>
      <td>${m.logloss.toFixed(3)}</td>
    </tr>`;
  }).join("");
}

function bindTabs() {
  document.querySelectorAll(".tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => {
        b.classList.toggle("is-active", b === button);
        b.setAttribute("aria-selected", b === button ? "true" : "false");
      });
      document.querySelectorAll(".panel").forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === button.dataset.tab);
      });
      if (button.dataset.tab === "performance") {
        setTimeout(drawPerformance, 40);
      }
    });
  });
}

function rocCurve(points) {
  const sorted = [...points].sort((a, b) => b.score - a.score);
  const positives = sorted.filter((p) => p.y === 1).length;
  const negatives = sorted.length - positives;
  let tp = 0, fp = 0;
  const xs = [0], ys = [0];
  for (const p of sorted) {
    if (p.y === 1) tp += 1; else fp += 1;
    xs.push(fp / negatives);
    ys.push(tp / positives);
  }
  return { x: xs, y: ys };
}

function calibrationCurve(points, bins = 8) {
  const sorted = [...points].sort((a, b) => a.score - b.score);
  const size = Math.ceil(sorted.length / bins);
  const xs = [], ys = [];
  for (let i = 0; i < sorted.length; i += size) {
    const chunk = sorted.slice(i, i + size);
    const meanScore = chunk.reduce((s, p) => s + p.score, 0) / chunk.length;
    const observed = chunk.reduce((s, p) => s + p.y, 0) / chunk.length;
    xs.push(meanScore); ys.push(observed);
  }
  return { x: xs, y: ys };
}

function drawPerformance() {
  if (!window.Plotly || !state.data) return;
  const calls = state.data.calls;
  const series = [
    ["pred_late_pre_text_ac", "#0f766e", "Late fusion (text + voice + pre-vol)"],
    ["pred_pre_vol", "#64748b", "Pre-call volatility"],
    ["pred_text", "#b45309", "Text only"],
    ["pred_voice", "#7c3aed", "Voice only"],
  ];
  const rocTraces = series.map(([field, color, name]) => {
    const curve = rocCurve(calls.map((c) => ({ y: c.high_vol ? 1 : 0, score: c[field] })));
    return { x: curve.x, y: curve.y, type: "scatter", mode: "lines", name, line: { color, width: 2 } };
  });
  rocTraces.push({ x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", line: { color: "#cbd5e1", dash: "dash", width: 1 }, showlegend: false, hoverinfo: "skip" });
  Plotly.react("rocChart", rocTraces, {
    ...baseLayout(),
    height: 300,
    margin: { l: 62, r: 20, t: 12, b: 48 },
    xaxis: { title: "False positive rate", range: [0, 1], gridcolor: "#eef2f7", zeroline: false },
    yaxis: { title: "True positive rate", range: [0, 1], gridcolor: "#eef2f7", zeroline: false },
    legend: { orientation: "h", y: -0.28, font: { size: 9 } },
  }, { displayModeBar: false, responsive: true });

  const calTraces = series.map(([field, color, name]) => {
    const curve = calibrationCurve(calls.map((c) => ({ y: c.high_vol ? 1 : 0, score: c[field] })));
    return { x: curve.x, y: curve.y, type: "scatter", mode: "lines+markers", name, line: { color, width: 2 }, marker: { size: 5 } };
  });
  calTraces.push({ x: [0, 1], y: [0, 1], type: "scatter", mode: "lines", line: { color: "#cbd5e1", dash: "dash", width: 1 }, showlegend: false, hoverinfo: "skip" });
  Plotly.react("calChart", calTraces, {
    ...baseLayout(),
    height: 300,
    margin: { l: 62, r: 20, t: 12, b: 48 },
    xaxis: { title: "Predicted risk", range: [0, 1], gridcolor: "#eef2f7", zeroline: false },
    yaxis: { title: "Observed high-vol rate", range: [0, 1], gridcolor: "#eef2f7", zeroline: false },
    legend: { orientation: "h", y: -0.28, font: { size: 9 } },
  }, { displayModeBar: false, responsive: true });

  const ordered = [...state.data.metrics].sort((a, b) => a.val_auc - b.val_auc);
  Plotly.react("ablationChart", [{
    x: ordered.map((m) => m.auc),
    y: ordered.map((m) => MODEL_NAMES[m.model] || m.model),
    type: "bar",
    orientation: "h",
    marker: { color: ordered.map((m) => m.model === "late_pre_text_ac" ? "#0f766e" : "#cbd5e1") },
    text: ordered.map((m) => m.auc.toFixed(3)),
    textposition: "outside",
    cliponaxis: false,
    hovertemplate: "%{y}<br>Test AUC %{x:.3f}<extra></extra>",
  }], {
    ...baseLayout(),
    height: 390,
    margin: { l: 265, r: 40, t: 8, b: 44 },
    xaxis: { title: "Test ROC-AUC (axis starts at 0.45)", range: [0.45, 0.86], gridcolor: "#eef2f7", zeroline: false },
    yaxis: { tickfont: { size: 10 } },
  }, { displayModeBar: false, responsive: true });
  state.performanceDrawn = true;
}

fetch("data.json")
  .then((response) => {
    if (!response.ok) throw new Error(`data.json: ${response.status}`);
    return response.json();
  })
  .then(init)
  .catch((error) => {
    console.error(error);
    document.querySelector("main").insertAdjacentHTML("afterbegin", '<p style="color:#b91c1c">Could not load dashboard data.</p>');
  });
