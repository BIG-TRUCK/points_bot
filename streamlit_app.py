"""Streamlit companion app for the CHL points bot.

Three views, switched via the `view` URL query param so they can be linked
to directly (`?view=score`, `?view=results`) as well as reachable by
clicking a tab in the nav bar rendered at the top of every view:
  - landing (default): the model report (same content as docs/index.html).
  - results (`?view=results`): browse the model's ranked unlabeled-card
    predictions and submit a point-value rating (opens a GitHub Issue —
    see github_feedback.py).
  - score (`?view=score`): type any card name and get a live rating,
    whether or not it's ever been played in CHL.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud: main file path = streamlit_app.py.
Secrets required (Settings -> Secrets in the Streamlit Cloud dashboard, or
.streamlit/secrets.toml locally — never commit that file):

    GITHUB_TOKEN = "ghp_..."          # fine-grained PAT, Issues: write, on this repo
    GITHUB_REPO  = "BIG-TRUCK/points_bot"

Data: app_data/{chl_model.pkl, chl_predictions.csv, chl_card_pool.pkl} — a
small snapshot committed alongside the app (unlike data/, which is
gitignored and too large to ship). Regenerate after every retrain with:

    python -m scripts.export_app_data
"""

import pickle
import urllib.parse

import pandas as pd
import streamlit as st

from github_feedback import feedback_configured, submit_feedback
from model.predict import VALID_POINTS, score_card
from report_content import REPORT_CSS, render_report_sections

APP_DATA_DIR = "app_data"
MODEL_PATH = f"{APP_DATA_DIR}/chl_model.pkl"
PREDICTIONS_PATH = f"{APP_DATA_DIR}/chl_predictions.csv"
CARD_POOL_PATH = f"{APP_DATA_DIR}/chl_card_pool.pkl"

st.set_page_config(page_title="CHL Points Bot", page_icon="🎯", layout="wide")


def _safe_str(value) -> str:
    return value if isinstance(value, str) else ""


def card_image_url(name: str) -> str:
    return f"https://api.scryfall.com/cards/named?exact={urllib.parse.quote(name)}&format=image&version=normal"


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_card_pool() -> pd.DataFrame:
    return pd.read_pickle(CARD_POOL_PATH)


@st.cache_resource
def load_model_artifact() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def feedback_widget(card_name: str, predicted_points, pointed_prob, key_prefix: str) -> None:
    state_key = f"{key_prefix}_submitted"
    if st.session_state.get(state_key):
        st.success("Feedback submitted — thank you!")
        return

    if not feedback_configured():
        st.caption("Feedback collection isn't configured on this deployment.")
        return

    with st.form(key=f"{key_prefix}_form"):
        cols = st.columns([2, 3, 1])
        your_points = cols[0].selectbox("Your rating", sorted(VALID_POINTS), key=f"{key_prefix}_points")
        note = cols[1].text_input("Note (optional)", key=f"{key_prefix}_note")
        submitted = cols[2].form_submit_button("Submit")
        if submitted:
            ok, msg = submit_feedback(
                card_name=card_name,
                your_points=your_points,
                predicted_points=predicted_points,
                pointed_prob=pointed_prob,
                note=note,
            )
            if ok:
                st.session_state[state_key] = True
                st.success(msg)
            else:
                st.error(msg)


NAV_TABS = [
    ("landing", "Overview"),
    ("score", "🃏 Try it out"),
    ("results", "🔎 See the results"),
]


def _flatten_for_markdown(text: str) -> str:
    # st.markdown runs unsafe_allow_html content through a CommonMark parser
    # first, which reads any 4+-space-indented line as an indented code
    # block — report_content.py's nested f-strings produce those freely.
    return "\n".join(line.lstrip() for line in text.split("\n"))


def nav_tabs(active: str) -> None:
    links = "".join(
        f'<a class="tab-link{" active" if key == active else ""}" href="?view={key}">{label}</a>'
        for key, label in NAV_TABS
    )
    body = f'{REPORT_CSS}<div class="viz-root"><div class="tab-row">{links}</div></div>'
    st.markdown(_flatten_for_markdown(body), unsafe_allow_html=True)


def render_results(predictions_df: pd.DataFrame, card_pool_df: pd.DataFrame) -> None:
    nav_tabs("results")
    st.subheader("Top suspects")
    st.write(
        "Unlabeled cards ranked by the model's estimated probability that they "
        "should be pointed. Tell us if you think the model's wrong."
    )

    card_oracle = card_pool_df.set_index("card_name")
    max_n = min(100, len(predictions_df))
    top_n = st.slider("How many to show", min_value=5, max_value=max_n, value=min(20, max_n), step=5)

    for _, row in predictions_df.head(top_n).iterrows():
        name = row["card_name"]
        with st.container(border=True):
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(card_image_url(name), width="stretch")
            with cols[1]:
                st.markdown(f"### {name}")
                if name in card_oracle.index:
                    card = card_oracle.loc[name]
                    mana = _safe_str(card.get("mana_cost"))
                    type_line = _safe_str(card.get("type_line"))
                    st.caption(f"{mana}  •  {type_line}")
                    oracle_text = _safe_str(card.get("oracle_text"))
                    if oracle_text:
                        st.write(oracle_text)
                st.metric("Pointed probability", f"{row['pointed_prob']:.1f}%")
                st.write(
                    f"Model estimate: **{int(row['est_points'])} pts**  |  "
                    f"Appearances: {int(row['appearances'])}  |  "
                    f"Avg placement: {row['avg_placement']:.1f}  |  "
                    f"Top-4 rate: {row['top4_rate']:.0%}"
                )
                feedback_widget(
                    card_name=name,
                    predicted_points=int(row["est_points"]),
                    pointed_prob=float(row["pointed_prob"]),
                    key_prefix=f"suspect_{name}",
                )


def render_score(card_pool_df: pd.DataFrame) -> None:
    nav_tabs("score")
    st.subheader("Score any card")
    st.write(
        "Type a card name to get the model's live rating. Cards that have "
        "never appeared in a CHL tournament decklist are scored from their "
        "printed properties alone — the model leans heavily on tournament "
        "performance, so ratings for never-played cards are less reliable."
    )

    card_name_input = st.text_input("Card name", placeholder="e.g. Sol Ring")
    go = st.button("Score this card", type="primary")

    if go and card_name_input.strip():
        with st.spinner(f"Scoring {card_name_input}..."):
            result = score_card(card_name_input.strip(), model_path=MODEL_PATH, card_pool=card_pool_df)

        if result is None:
            st.error(
                f"Couldn't find '{card_name_input}' — check the spelling "
                "(Scryfall fuzzy-matches, but it still needs to be a real card)."
            )
        else:
            name = result["card_name"]
            st.markdown(f"## {name}")
            cols = st.columns([1, 3])
            with cols[0]:
                st.image(card_image_url(name), width="stretch")
            with cols[1]:
                mana = _safe_str(result.get("mana_cost"))
                type_line = _safe_str(result.get("type_line"))
                st.caption(f"{mana}  •  {type_line}")
                oracle_text = _safe_str(result.get("oracle_text"))
                if oracle_text:
                    st.write(oracle_text)

                if not result["has_tournament_history"]:
                    st.warning(
                        "This card has no CHL tournament history — rating is based "
                        "on card properties + oracle text only."
                    )

                if result["already_pointed"]:
                    st.info(f"Already on the points list at **{result['points']} pts**.")
                else:
                    m1, m2 = st.columns(2)
                    m1.metric("Pointed probability", f"{result['pointed_prob']:.1f}%")
                    m2.metric("Estimated points", f"{result['est_points']} pts")

            if not result["already_pointed"]:
                st.divider()
                feedback_widget(
                    card_name=name,
                    predicted_points=result["est_points"],
                    pointed_prob=result["pointed_prob"],
                    key_prefix=f"lookup_{name}",
                )


def render_landing(artifact: dict, predictions_df: pd.DataFrame, card_pool_df: pd.DataFrame) -> None:
    nav_tabs("landing")
    # Relative hrefs (?view=...) resolve against whatever domain the app is
    # actually running on — localhost during dev, the real Streamlit Cloud
    # URL once deployed — so these work correctly without hardcoding it.
    sections_html = render_report_sections(artifact, predictions_df, card_pool_df)
    body = f"""{REPORT_CSS}
<div class="viz-root">
  <p class="subtitle">10-point Canadian Highlander (CHL) points model — evaluation summary.</p>
  {sections_html}
</div>"""
    st.markdown(_flatten_for_markdown(body), unsafe_allow_html=True)


st.title("🎯 CHL Points Bot")
st.caption("Community feedback tool for the 10-point Canadian Highlander (CHL) points model.")

try:
    predictions_df = load_predictions()
    card_pool_df = load_card_pool()
    model_artifact = load_model_artifact()
except FileNotFoundError as e:
    st.error(
        f"Missing bundled data: {e}\n\n"
        "Run `python -m scripts.export_app_data` after training and commit the "
        "resulting app_data/ files."
    )
    st.stop()

view = st.query_params.get("view", "landing")

if view == "results":
    render_results(predictions_df, card_pool_df)
elif view == "score":
    render_score(card_pool_df)
else:
    render_landing(model_artifact, predictions_df, card_pool_df)
