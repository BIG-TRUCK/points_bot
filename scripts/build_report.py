"""Builds a static GitHub Pages report summarizing the latest trained model.

Reads data/chl_model.pkl (the full training artifact — SHAP, LOOCV results,
classifier_eval) and data/chl_predictions.csv, and writes a single
self-contained docs/index.html: dataset size, which model candidates were
evaluated and which won each stage, AUPRC/F1 (classifier) and MAE
(regressor), SHAP feature-importance bar charts for every candidate model,
and the top-15 suspect cards.

Enable Pages once in the repo's Settings -> Pages -> "Deploy from a branch",
source = main, folder = /docs. Re-run this after every retrain:

    python -m model.train
    python -m model.predict
    python -m scripts.build_report
"""

import html
import logging
import os
import pickle
from datetime import datetime, timezone
from glob import glob

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = "data"
MODEL_PATH = os.path.join(DATA_DIR, "chl_model.pkl")
PREDICTIONS_PATH = os.path.join(DATA_DIR, "chl_predictions.csv")
DOCS_DIR = "docs"
OUTPUT_PATH = os.path.join(DOCS_DIR, "index.html")

TOP_N_SUSPECTS = 15
TOP_N_SHAP = 15


def _latest_per_card_df() -> pd.DataFrame:
    candidates = sorted(glob(os.path.join(DATA_DIR, "chl_dataset_per_card_*.pkl")))
    if not candidates:
        raise FileNotFoundError("No per_card dataset found. Run pipeline.py first.")
    with open(candidates[-1], "rb") as f:
        return pickle.load(f)


def _esc(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _bar_chart_html(series: "pd.Series | None", empty_note: str) -> str:
    """Renders a horizontal bar chart (mean |SHAP|, single sequential hue)
    as plain HTML/CSS — no JS, so it works unmodified on GitHub Pages."""
    if series is None or series.empty:
        return f'<p class="muted">{_esc(empty_note)}</p>'

    top = series.head(TOP_N_SHAP)
    max_val = float(top.max()) or 1.0
    rows = []
    for name, val in top.items():
        pct = max(2.0, (float(val) / max_val) * 100)
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-label" title="{_esc(name)}">{_esc(name)}</div>
          <div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%" title="{_esc(name)}: {val:.4f}"></div></div>
          <div class="bar-value">{val:.3f}</div>
        </div>""")
    return f'<div class="bar-chart">{"".join(rows)}</div>'


def _shap_section_html(shap_by_model: "dict[str, pd.DataFrame] | None", best_name: str, stage_label: str) -> str:
    """Renders one SHAP panel per candidate model, in a responsive grid —
    every candidate gets a chart now, not just the stage's winner."""
    if not shap_by_model:
        return f'<p class="muted">No SHAP data available for the {_esc(stage_label)} stage.</p>'

    panels = []
    for name, df in shap_by_model.items():
        badge = ' <span class="badge">winner</span>' if name == best_name else ""
        series = df["mean_abs_shap"] if df is not None else None
        panels.append(f"""
      <div class="shap-panel">
        <p class="panel-title">{_esc(name)}{badge}</p>
        {_bar_chart_html(series, f"No SHAP values for {name}.")}
      </div>""")
    return f'<div class="shap-grid">{"".join(panels)}</div>'


def _suspects_table_html(df: pd.DataFrame) -> str:
    top = df.head(TOP_N_SUSPECTS)
    rows = []
    for _, row in top.iterrows():
        rows.append(f"""
        <tr>
          <td>{_esc(row['card_name'])}</td>
          <td class="num">{float(row['pointed_prob']):.1f}%</td>
          <td class="num">{int(row['est_points'])}</td>
          <td class="num">{int(row['appearances'])}</td>
          <td class="num">{float(row['avg_placement']):.1f}</td>
          <td class="num">{float(row['top4_rate']):.0%}</td>
        </tr>""")
    return "".join(rows)


def _stat_tile(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="stat-note">{_esc(note)}</div>' if note else ""
    return f"""
    <div class="stat-tile">
      <div class="stat-label">{_esc(label)}</div>
      <div class="stat-value">{value}</div>
      {note_html}
    </div>"""


def build(model_path: str = MODEL_PATH, predictions_path: str = PREDICTIONS_PATH) -> None:
    logging.basicConfig(level=logging.INFO)

    with open(model_path, "rb") as f:
        artifact = pickle.load(f)
    predictions = pd.read_csv(predictions_path)
    per_card = _latest_per_card_df()

    n_total = len(per_card)
    n_pointed = int((per_card["points"].notna() & (per_card["points"] > 0)).sum())
    n_unpointed = n_total - n_pointed

    clf_results = artifact.get("clf_results", {})
    reg_results = artifact.get("reg_results", {})
    best_clf_name = artifact.get("best_clf_name", "?")
    best_reg_name = artifact.get("best_reg_name", "?")
    clf_eval = artifact.get("classifier_eval", {})
    naive_mae = artifact.get("naive_baseline_mae")
    reg_mae = reg_results.get(best_reg_name, {}).get("mae")

    clf_candidates_html = "".join(
        f"<li><strong>{_esc(name)}</strong> — mean rank pct {res['mean_rank_pct']:.3f}"
        f"{' (winner)' if name == best_clf_name else ''}</li>"
        for name, res in clf_results.items()
    )
    reg_candidates_html = "".join(
        f"<li><strong>{_esc(name)}</strong> — MAE {res['mae']:.3f}"
        f"{' (winner)' if name == best_reg_name else ''}</li>"
        for name, res in reg_results.items()
    )

    shap_clf_html = _shap_section_html(artifact.get("shap_importance_clf"), best_clf_name, "classifier")
    shap_reg_html = _shap_section_html(artifact.get("shap_importance_reg"), best_reg_name, "regressor")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHL Points Bot — Model Report</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: #fcfcfb;
    --surface-page: #f9f9f7;
    --text-primary: #0b0b0b;
    --text-secondary: #52514e;
    --text-muted: #898781;
    --gridline: #e1e0d9;
    --border: rgba(11,11,11,0.10);
    --series-1: #2a78d6;
    --series-1-track: #e8f0fb;
  }}
  @media (prefers-color-scheme: dark) {{
    .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --surface-page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --border: rgba(255,255,255,0.10);
      --series-1: #3987e5;
      --series-1-track: #1f2c3d;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body.viz-root {{
    margin: 0;
    background: var(--surface-page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5;
  }}
  main {{ max-width: 960px; margin: 0 auto; padding: 32px 20px 64px; }}
  h1 {{ font-size: 28px; margin: 0 0 4px; }}
  h2 {{ font-size: 18px; margin: 40px 0 12px; padding-top: 8px; border-top: 1px solid var(--gridline); }}
  .subtitle {{ color: var(--text-secondary); margin: 0 0 24px; }}
  .muted {{ color: var(--text-muted); font-size: 14px; }}
  .card {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
  }}
  .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
  .stat-tile {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
  }}
  .stat-label {{ font-size: 13px; color: var(--text-secondary); }}
  .stat-value {{ font-size: 30px; font-weight: 600; margin-top: 4px; font-variant-numeric: proportional-nums; }}
  .stat-note {{ font-size: 12px; color: var(--text-muted); margin-top: 4px; }}
  ul.candidates {{ margin: 0; padding-left: 20px; color: var(--text-secondary); }}
  ul.candidates li {{ margin: 4px 0; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
  @media (max-width: 720px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .shap-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 720px) {{ .shap-grid {{ grid-template-columns: 1fr; }} }}
  .shap-panel {{
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
  }}
  .panel-title {{ font-weight: 600; margin: 0 0 10px; }}
  .badge {{
    font-size: 11px; font-weight: 500; color: var(--text-secondary);
    border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px;
    text-transform: uppercase; letter-spacing: 0.02em;
  }}
  .bar-chart {{ display: flex; flex-direction: column; gap: 6px; }}
  .bar-row {{ display: grid; grid-template-columns: 140px 1fr 56px; align-items: center; gap: 10px; }}
  .bar-label {{
    font-size: 12px; color: var(--text-secondary); text-align: right;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .bar-track {{ background: var(--series-1-track); border-radius: 4px; height: 16px; }}
  .bar-fill {{ background: var(--series-1); height: 16px; border-radius: 0 4px 4px 0; min-width: 4px; }}
  .bar-value {{ font-size: 12px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--gridline); text-align: left; }}
  th {{ color: var(--text-secondary); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.02em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  footer {{ margin-top: 48px; color: var(--text-muted); font-size: 13px; }}
  a {{ color: var(--series-1); }}
</style>
</head>
<body class="viz-root">
<main>
  <h1>🎯 CHL Points Bot — Model Report</h1>
  <p class="subtitle">7-point Highlander points model — evaluation summary. Generated {generated_at}.</p>

  <section class="card">
    <h2 style="margin-top:0; border-top:none; padding-top:0;">What's evaluated</h2>
    <p>
      <strong>{n_total:,}</strong> unique cards tracked from CHL tournament decklists —
      <strong>{n_pointed}</strong> currently pointed, <strong>{n_unpointed:,}</strong> unlabeled.
      The model is two-stage: a <strong>classifier</strong> (should this card be pointed at all?)
      trained on all cards, and a <strong>regressor</strong> (how many points?) trained only on
      pointed cards. Both stages pick the best of several candidate models via
      leave-one-out cross-validation.
    </p>
    <div class="two-col">
      <div>
        <p><strong>Classifier candidates</strong></p>
        <ul class="candidates">{clf_candidates_html}</ul>
      </div>
      <div>
        <p><strong>Regressor candidates</strong></p>
        <ul class="candidates">{reg_candidates_html}</ul>
      </div>
    </div>
  </section>

  <h2>Classifier performance — {_esc(best_clf_name)}</h2>
  <div class="stat-grid">
    {_stat_tile("AUPRC", f"{clf_eval.get('auprc', 0):.3f}", "headline metric — robust to the ~40:1,584 class imbalance")}
    {_stat_tile("Best F1", f"{clf_eval.get('best_f1', 0):.3f}", f"at threshold {clf_eval.get('best_f1_threshold', 0):.2f}")}
    {_stat_tile("F1 @ 0.5 threshold", f"{clf_eval.get('f1_at_0.5', 0):.3f}")}
    {_stat_tile("Precision @ 0.5", f"{clf_eval.get('precision_at_0.5', 0):.3f}")}
    {_stat_tile("Recall @ 0.5", f"{clf_eval.get('recall_at_0.5', 0):.3f}")}
    {_stat_tile("AUROC", f"{clf_eval.get('auroc', 0):.3f}", "reference only — reads high under this imbalance")}
  </div>
  <p class="muted">{_esc(clf_eval.get("note", ""))}</p>

  <h2>Regressor performance — {_esc(best_reg_name)}</h2>
  <div class="stat-grid">
    {_stat_tile("MAE (points)", f"{reg_mae:.3f}" if reg_mae is not None else "—")}
    {_stat_tile("Naive baseline MAE", f"{naive_mae:.3f}" if naive_mae is not None else "—", "always predicting the mode")}
  </div>

  <h2>Feature importance (mean |SHAP|) — classifier</h2>
  <p class="muted">Every candidate model, not just the winner. SVM/OrdinalRidge have no closed-form SHAP, so they're computed via KernelExplainer on a subsample — treat those as directional, not exact.</p>
  {shap_clf_html}

  <h2>Feature importance (mean |SHAP|) — regressor</h2>
  {shap_reg_html}

  <h2>Top {TOP_N_SUSPECTS} suspects</h2>
  <p class="muted">Unlabeled cards ranked by the classifier's estimated pointed-probability.</p>
  <table>
    <thead>
      <tr>
        <th>Card</th>
        <th class="num">Pointed prob.</th>
        <th class="num">Est. points</th>
        <th class="num">Appearances</th>
        <th class="num">Avg placement</th>
        <th class="num">Top-4 rate</th>
      </tr>
    </thead>
    <tbody>
      {_suspects_table_html(predictions)}
    </tbody>
  </table>

  <footer>
    Generated by <code>scripts/build_report.py</code> from <code>data/chl_model.pkl</code> and
    <code>data/chl_predictions.csv</code>. Source: <a href="https://github.com/BIG-TRUCK/points_bot">BIG-TRUCK/points_bot</a>.
    Rate any of these cards yourself in the <a href="https://share.streamlit.io/">companion Streamlit app</a>.
  </footer>
</main>
</body>
</html>
"""

    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_doc)
    logger.info(f"Report written to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
