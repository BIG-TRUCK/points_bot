"""Shared HTML for the model report - used by both:
  - scripts/build_report.py, which wraps this in a full page for the static
    GitHub Pages report (docs/index.html)
  - streamlit_app.py, which embeds it as the app's landing page

Kept in one place so both surfaces always show identical numbers/charts;
each caller only supplies its own page-level chrome (title, footer, nav).
"""

import html
from typing import Optional

import pandas as pd

from model.embedding_probe import probe_dimension

TOP_N_SUSPECTS = 15
TOP_N_SHAP = 15

# Scoped to .viz-root and its descendants - no `body` selector, since the
# Streamlit embedding doesn't control the actual <body> tag.
REPORT_CSS = """<style>
  .viz-root {
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
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5;
    color: var(--text-primary);
  }
  @media (prefers-color-scheme: dark) {
    .viz-root {
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
    }
  }
  .viz-root * { box-sizing: border-box; }
  .viz-root h1 { font-size: 28px; margin: 0 0 4px; }
  .viz-root h2 { font-size: 18px; margin: 40px 0 12px; padding-top: 8px; border-top: 1px solid var(--gridline); }
  .viz-root .subtitle { color: var(--text-secondary); margin: 0 0 24px; }
  .viz-root .muted { color: var(--text-muted); font-size: 14px; }
  .viz-root .card {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 24px;
  }
  .viz-root .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .viz-root .stat-tile {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 18px;
  }
  .viz-root .stat-label { font-size: 13px; color: var(--text-secondary); }
  .viz-root .stat-value { font-size: 30px; font-weight: 600; margin-top: 4px; font-variant-numeric: proportional-nums; }
  .viz-root .stat-note { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
  .viz-root .stat-subs { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--gridline); }
  .viz-root .stat-sub { font-size: 12px; }
  .viz-root .stat-sub-label { color: var(--text-muted); margin-right: 4px; }
  .viz-root .stat-sub-value { color: var(--text-secondary); font-weight: 600; font-variant-numeric: tabular-nums; }
  .viz-root ul.candidates { margin: 0; padding-left: 20px; color: var(--text-secondary); }
  .viz-root ul.candidates li { margin: 4px 0; }
  .viz-root ul.candidates .params { font-size: 11px; color: var(--text-muted); margin: 2px 0 8px; }
  .viz-root .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 720px) { .viz-root .two-col { grid-template-columns: 1fr; } }
  .viz-root .shap-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 720px) { .viz-root .shap-grid { grid-template-columns: 1fr; } }
  .viz-root .shap-panel {
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
  }
  .viz-root .panel-title { font-weight: 600; margin: 0 0 10px; }
  .viz-root .badge {
    font-size: 11px; font-weight: 500; color: var(--text-secondary);
    border: 1px solid var(--border); border-radius: 999px; padding: 1px 8px;
    text-transform: uppercase; letter-spacing: 0.02em;
  }
  .viz-root .bar-chart { display: flex; flex-direction: column; gap: 6px; }
  .viz-root .bar-row { display: grid; grid-template-columns: 140px 1fr 56px; align-items: center; gap: 10px; }
  .viz-root .bar-label {
    font-size: 12px; color: var(--text-secondary); text-align: right;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .viz-root .bar-track { background: var(--series-1-track); border-radius: 4px; height: 16px; }
  .viz-root .bar-fill { background: var(--series-1); height: 16px; border-radius: 0 4px 4px 0; min-width: 4px; }
  .viz-root .bar-value { font-size: 12px; color: var(--text-secondary); font-variant-numeric: tabular-nums; }
  .viz-root .emb-probe { font-size: 11px; margin: -2px 0 4px 150px; }
  .viz-root .emb-probe span { margin-right: 12px; }
  .viz-root .emb-probe .emb-pos { color: var(--series-1); }
  .viz-root .emb-probe .emb-neg { color: var(--text-muted); }
  .viz-root table { width: 100%; border-collapse: collapse; font-size: 14px; }
  .viz-root th, .viz-root td { padding: 8px 10px; border-bottom: 1px solid var(--gridline); text-align: left; }
  .viz-root th { color: var(--text-secondary); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.02em; }
  .viz-root td.num, .viz-root th.num { text-align: right; font-variant-numeric: tabular-nums; }
  .viz-root a { color: var(--series-1); }
</style>"""


def _esc(value) -> str:
    return html.escape(str(value)) if value is not None else ""


def _embedding_probe_html(feature_name: str, per_card: "Optional[pd.DataFrame]") -> str:
    """For a tags_emb_* SHAP feature, a compact caption of which real card
    tags associate with high/low values on that dimension - see
    model/embedding_probe.py. Empty string for any other feature, or if
    per_card wasn't supplied."""
    if per_card is None:
        return ""
    result = probe_dimension(per_card, feature_name)
    if not result:
        return ""

    spans = []
    if result["positive"]:
        tags = ", ".join(_esc(tag) for tag, _delta in result["positive"])
        spans.append(f'<span class="emb-pos">&#8593; {tags}</span>')
    if result["negative"]:
        tags = ", ".join(_esc(tag) for tag, _delta in result["negative"])
        spans.append(f'<span class="emb-neg">&#8595; {tags}</span>')
    if not spans:
        return ""
    return f'<div class="emb-probe">{"".join(spans)}</div>'


def _bar_chart_html(
    series: "Optional[pd.Series]", empty_note: str, per_card: "Optional[pd.DataFrame]" = None
) -> str:
    """Renders a horizontal bar chart (mean |SHAP|, single sequential hue)
    as plain HTML/CSS - no JS required. For tags_emb_* features, adds a
    caption of the real tags that dimension correlates with (see
    _embedding_probe_html) - those dimensions have no inherent meaning the
    way a TF-IDF or structured-attribute feature name does."""
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
        </div>
        {_embedding_probe_html(name, per_card)}""")
    return f'<div class="bar-chart">{"".join(rows)}</div>'


def _shap_section_html(
    shap_by_model: "Optional[dict[str, pd.DataFrame]]",
    best_name: str,
    stage_label: str,
    per_card: "Optional[pd.DataFrame]" = None,
) -> str:
    """Renders one SHAP panel per candidate model, in a responsive grid -
    every candidate gets a chart, not just the stage's winner."""
    if not shap_by_model:
        return f'<p class="muted">No SHAP data available for the {_esc(stage_label)} stage.</p>'

    panels = []
    for name, df in shap_by_model.items():
        badge = ' <span class="badge">winner</span>' if name == best_name else ""
        series = df["mean_abs_shap"] if df is not None else None
        panels.append(f"""
      <div class="shap-panel">
        <p class="panel-title">{_esc(name)}{badge}</p>
        {_bar_chart_html(series, f"No SHAP values for {name}.", per_card=per_card)}
      </div>""")
    return f'<div class="shap-grid">{"".join(panels)}</div>'


def _suspects_table_html(df: pd.DataFrame, top_n: int) -> str:
    top = df.head(top_n)
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


def _format_params(params: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(params.items()))


def _stat_tile(label: str, value: str, note: str = "", subs: "Optional[list[tuple[str, str]]]" = None) -> str:
    note_html = f'<div class="stat-note">{_esc(note)}</div>' if note else ""
    subs_html = ""
    if subs:
        sub_items = "".join(
            f'<div class="stat-sub"><span class="stat-sub-label">{_esc(sub_label)}</span>'
            f'<span class="stat-sub-value">{_esc(sub_value)}</span></div>'
            for sub_label, sub_value in subs
        )
        subs_html = f'<div class="stat-subs">{sub_items}</div>'
    return f"""
    <div class="stat-tile">
      <div class="stat-label">{_esc(label)}</div>
      <div class="stat-value">{value}</div>
      {note_html}
      {subs_html}
    </div>"""


def render_report_sections(artifact: dict, predictions: pd.DataFrame, per_card: pd.DataFrame) -> str:
    """Returns the report body - "What's evaluated" through the top-suspects
    table - with no page-level title, CSS, or footer, so callers can wrap it
    however suits their context (standalone page vs. embedded landing page).
    """
    n_total = len(per_card)
    n_pointed = int((per_card["points"].notna() & (per_card["points"] > 0)).sum())
    n_unpointed = n_total - n_pointed
    # F1 of a naive "flag every card as pointed" classifier: precision =
    # base rate, recall = 1.0, so F1 = 2*base_rate/(1+base_rate), which
    # simplifies to this - a more meaningful yardstick than an arbitrary
    # fixed number under this ~40:1,584 imbalance (see AUPRC's note below).
    naive_clf_f1 = (2 * n_pointed / (n_pointed + n_total)) if n_total else 0.0

    clf_results = artifact.get("clf_results", {})
    reg_results = artifact.get("reg_results", {})
    best_clf_name = artifact.get("best_clf_name", "?")
    best_reg_name = artifact.get("best_reg_name", "?")
    clf_eval = artifact.get("classifier_eval", {})
    naive_mae = artifact.get("naive_baseline_mae")
    reg_mae = reg_results.get(best_reg_name, {}).get("mae")
    clf_tuned_params = artifact.get("clf_tuned_params", {})
    reg_tuned_params = artifact.get("reg_tuned_params", {})

    clf_candidates_html = "".join(
        f"<li><strong>{_esc(name)}</strong> - mean rank pct {res['mean_rank_pct']:.3f}"
        f"{' (winner)' if name == best_clf_name else ''}"
        + (
            f"<div class='params'>{_esc(_format_params(params))}</div>"
            if (params := clf_tuned_params.get(name))
            else ""
        )
        + "</li>"
        for name, res in clf_results.items()
    )
    reg_candidates_html = "".join(
        f"<li><strong>{_esc(name)}</strong> - MAE {res['mae']:.3f}"
        f"{' (winner)' if name == best_reg_name else ''}"
        + (
            f"<div class='params'>{_esc(_format_params(params))}</div>"
            if (params := reg_tuned_params.get(name))
            else ""
        )
        + "</li>"
        for name, res in reg_results.items()
    )

    shap_clf_html = _shap_section_html(
        artifact.get("shap_importance_clf"), best_clf_name, "classifier", per_card=per_card
    )
    shap_reg_html = _shap_section_html(
        artifact.get("shap_importance_reg"), best_reg_name, "regressor", per_card=per_card
    )

    return f"""
  <section class="card">
    <h2 style="margin-top:0; border-top:none; padding-top:0;">What's evaluated</h2>
    <p>
      <strong>{n_total:,}</strong> unique cards tracked from CHL tournament decklists -
      <strong>{n_pointed}</strong> currently pointed, <strong>{n_unpointed:,}</strong> unlabeled.
      The model is two-stage: a <strong>classifier</strong> (should this card be pointed at all?)
      trained on all cards, and a <strong>regressor</strong> (how many points?) trained only on
      pointed cards. Both stages pick the best of several candidate models via
      leave-one-out cross-validation; each candidate's hyperparameters (shown below it)
      come from a small random search on a cheaper K-fold split beforehand.
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

  <h2>Classifier performance - {_esc(best_clf_name)}</h2>
  <div class="stat-grid">
    {_stat_tile("AUPRC", f"{clf_eval.get('auprc', 0):.3f}")}
    {_stat_tile(
        "Best F1", f"{clf_eval.get('best_f1', 0):.3f}", f"at threshold {clf_eval.get('best_f1_threshold', 0):.2f}",
        subs=[
            ("F1 @ 0.5", f"{clf_eval.get('f1_at_0.5', 0):.3f}"),
            ("Precision @ 0.5", f"{clf_eval.get('precision_at_0.5', 0):.3f}"),
            ("Recall @ 0.5", f"{clf_eval.get('recall_at_0.5', 0):.3f}"),
        ],
    )}
    {_stat_tile("Naive baseline F1", f"{naive_clf_f1:.3f}")}
    {_stat_tile("AUROC", f"{clf_eval.get('auroc', 0):.3f}", "reference only - data too imbalanced for this to be meaningful")}
  </div>
  <p class="muted">{_esc(clf_eval.get("note", ""))}</p>

  <h2>Regressor performance - {_esc(best_reg_name)}</h2>
  <div class="stat-grid">
    {_stat_tile("MAE (points)", f"{reg_mae:.3f}" if reg_mae is not None else "-")}
    {_stat_tile("Naive baseline MAE", f"{naive_mae:.3f}" if naive_mae is not None else "-")}
  </div>

  <h2>Feature importance (mean |SHAP|) - classifier</h2>
  <p class="muted">SVM/OrdinalRidge have no closed-form SHAP, so they're computed via KernelExplainer on a subsample (treat those as directional only). For tags_emb_* features, &#8593;/&#8595; show which real card tags average highest/lowest on that dimension</p>
  {shap_clf_html}

  <h2>Feature importance (mean |SHAP|) - regressor</h2>
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
      {_suspects_table_html(predictions, TOP_N_SUSPECTS)}
    </tbody>
  </table>
"""
