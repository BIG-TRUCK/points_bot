"""Posts visitor feedback from the Streamlit app as GitHub Issues.

Used instead of a local file so feedback submitted by visitors to the hosted
app survives redeploys/restarts (Streamlit Community Cloud's filesystem does
not persist across those). Each submission opens one Issue, labeled
"feedback", with a small JSON block in the body so it can later be parsed
back into data/feedback.json's format for training (see model/feedback.py).

Requires two Streamlit secrets (Settings -> Secrets in Streamlit Cloud, or
.streamlit/secrets.toml locally — never commit that file):

    GITHUB_TOKEN = "ghp_..."          # a fine-grained PAT with Issues: write
    GITHUB_REPO  = "BIG-TRUCK/points_bot"
"""

import json
import logging
from typing import Optional

import requests
import streamlit as st

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"


def feedback_configured() -> bool:
    """Whether GITHUB_TOKEN/GITHUB_REPO secrets are present."""
    return bool(st.secrets.get("GITHUB_TOKEN")) and bool(st.secrets.get("GITHUB_REPO"))


def submit_feedback(
    card_name: str,
    your_points: int,
    predicted_points: Optional[int] = None,
    pointed_prob: Optional[float] = None,
    note: str = "",
) -> tuple[bool, str]:
    """Opens a GitHub Issue recording a visitor's point-value feedback.

    Returns:
        (success, message) — message is a short human-readable status,
        either a confirmation with the issue URL or an error to display.
    """
    if not feedback_configured():
        return False, "Feedback isn't configured on this deployment (missing GITHUB_TOKEN/GITHUB_REPO secrets)."

    token = st.secrets["GITHUB_TOKEN"]
    repo = st.secrets["GITHUB_REPO"]

    payload = {
        "card_name": card_name,
        "points": your_points,
        "predicted": predicted_points,
        "pointed_prob": pointed_prob,
    }
    body_lines = [
        f"**Card:** {card_name}",
        f"**Visitor rating:** {your_points} pts",
    ]
    if predicted_points is not None:
        body_lines.append(f"**Model estimate:** {predicted_points} pts" + (
            f" ({pointed_prob:.1f}% pointed-probability)" if pointed_prob is not None else ""
        ))
    if note:
        body_lines.append(f"**Note:** {note}")
    body_lines += ["", "```json", json.dumps(payload, indent=2), "```"]

    try:
        resp = requests.post(
            f"{GITHUB_API}/repos/{repo}/issues",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "title": f"[feedback] {card_name}: {your_points}pt",
                "body": "\n".join(body_lines),
                "labels": ["feedback"],
            },
            timeout=10,
        )
    except requests.RequestException as e:
        logger.warning(f"GitHub feedback submission failed for '{card_name}': {e}")
        return False, f"Couldn't reach GitHub: {e}"

    if resp.status_code == 201:
        issue_url = resp.json().get("html_url", "")
        return True, f"Feedback recorded. Thanks! {issue_url}"

    logger.warning(f"GitHub Issues API returned {resp.status_code}: {resp.text[:300]}")
    return False, f"GitHub API error ({resp.status_code}). Please try again later."
