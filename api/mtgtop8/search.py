import json
import logging
import os
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://mtgtop8.com"
SEARCH_URL = (
    f"{BASE_URL}/search"
    "?format=CHL"
    "&compet_check%5BP%5D=1"
    "&compet_check%5BM%5D=1"
    "&compet_check%5BC%5D=1"
)
OUTPUT_DIR = os.path.join("data", "CHL")
HEADERS = {"User-Agent": "BIGTRUCKPointsBot/1.0"}
REQUEST_DELAY = 0.5  # seconds between requests


def _get(url: str) -> Optional[requests.Response]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        return response
    except Exception as e:
        logger.error(f"Request failed for {url}: {e}")
        return None


def _parse_placement(text: str) -> Optional[int]:
    """Returns the best (lowest) rank from a string like '1', '3-4', or '5-8'."""
    match = re.match(r"(\d+)", text.strip())
    return int(match.group(1)) if match else None


def _parse_date(text: str) -> Optional[str]:
    """Parses a DD/MM/YY date string and returns 'MM/YYYY'."""
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", text)
    if not match:
        return None
    _, month, year = match.groups()
    year = f"20{year}" if len(year) == 2 else year
    return f"{int(month):02d}/{year}"


def get_deck_results(soup: BeautifulSoup) -> list[dict]:
    """Parses search results rows into dicts with url, name, placement, date, and level."""
    results = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if "event?" not in href or "d=" not in href:
            continue
        url = href if href.startswith("http") else f"{BASE_URL}/{href}"
        if url in seen:
            continue
        seen.add(url)

        name = a.get_text(strip=True)
        tr = a.find_parent("tr")
        if not tr:
            results.append({"url": url, "name": name, "placement": None, "date": None, "level": None})
            continue

        placement: Optional[int] = None
        date: Optional[str] = None
        level: Optional[int] = None

        for td in tr.find_all("td"):
            td_text = td.get_text(strip=True)

            if placement is None and re.fullmatch(r"\d+(-\d+)?", td_text):
                placement = _parse_placement(td_text)
            elif date is None and re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", td_text):
                date = _parse_date(td_text)

            stars = [img for img in td.find_all("img") if "star.png" in str(img.get("src", ""))]
            if stars:
                level = len(stars)

        results.append({"url": url, "name": name, "placement": placement, "date": date, "level": level})

    return results

def get_mtgo_download_url(deck_url: str) -> Optional[str]:
    """Fetches a deck page and returns the MTGO .txt download URL."""
    response = _get(deck_url)
    if not response:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    for a in soup.find_all("a", href=True):
        if "MTGO" in a.get_text():
            href = str(a["href"])
            return href if href.startswith("http") else f"{BASE_URL}/{href}"
    return None


def _save_metadata(deck_id: str, name: str, placement: Optional[int], date: Optional[str], level: Optional[int]) -> None:
    """Writes deck metadata to data/CHL/<deck_id>.json."""
    filepath = os.path.join(OUTPUT_DIR, f"{deck_id}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump({"name": name, "placement": placement, "date": date, "level": level}, f, indent=2)


def _download_decklist(url: str, deck_id: str) -> bool:
    """Downloads a decklist .txt file into data/CHL/<deck_id>.txt."""
    filepath = os.path.join(OUTPUT_DIR, f"{deck_id}.txt")
    if os.path.exists(filepath):
        logger.info(f"Already downloaded, skipping: {deck_id}.txt")
        return True
    response = _get(url)
    if not response:
        return False
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(response.text)
    logger.info(f"Saved: {deck_id}.txt")
    return True


def scrape_all() -> None:
    """Scrapes all CHL tournament decklists from mtgtop8.

    For each result, saves:
      - data/CHL/<deck_id>.txt  — the MTGO decklist
      - data/CHL/<deck_id>.json — metadata (name, placement, date, level)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    page = 1

    while True:
        url = SEARCH_URL + f"&current_page={page}"
        logger.info(f"Fetching search results page {page}: {url}")
        response = _get(url)
        if not response:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        deck_results = get_deck_results(soup)
        logger.info(f"Found {len(deck_results)} deck(s) on page {page}")

        if not deck_results:
            break

        for result in deck_results:
            deck_url = result["url"]
            deck_id = deck_url.split("d=")[-1].split("&")[0]

            mtgo_url = get_mtgo_download_url(deck_url)
            time.sleep(REQUEST_DELAY)

            if mtgo_url:
                _download_decklist(mtgo_url, deck_id)
                time.sleep(REQUEST_DELAY)
            else:
                logger.warning(f"No MTGO link found for deck {deck_id} ({deck_url})")

            _save_metadata(
                deck_id,
                name=result["name"],
                placement=result["placement"],
                date=result["date"],
                level=result["level"],
            )

        page += 1
        if current_url:
            time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scrape_all()
