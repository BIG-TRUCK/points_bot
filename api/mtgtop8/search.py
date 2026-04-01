import logging
import os
import time
import requests
from bs4 import BeautifulSoup
from typing import Optional

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


def get_deck_links(soup: BeautifulSoup) -> list[str]:
    """Extracts unique deck page URLs from a search results page."""
    seen = set()
    links = []
    for a in soup.find_all("a", href=True):
        href = str(a["href"])
        if "event?" in href and "d=" in href:
            url = href if href.startswith("http") else f"{BASE_URL}/{href}"
            if url not in seen:
                seen.add(url)
                links.append(url)
    return links


def get_next_page_url(soup: BeautifulSoup) -> Optional[str]:
    """Returns the URL of the next search results page, or None if on the last page."""
    for a in soup.find_all("a", href=True):
        if "next" in a.get_text(strip=True).lower():
            href = str(a["href"])
            return href if href.startswith("http") else f"{BASE_URL}/{href}"
    return None


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


def download_decklist(url: str, deck_id: str) -> bool:
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
    """Scrapes all CHL tournament decklists from mtgtop8 and saves MTGO .txt files to data/CHL/."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    current_url: Optional[str] = SEARCH_URL
    page = 1

    while current_url:
        logger.info(f"Fetching search results page {page}: {current_url}")
        response = _get(current_url)
        if not response:
            break

        soup = BeautifulSoup(response.text, "html.parser")
        deck_links = get_deck_links(soup)
        logger.info(f"Found {len(deck_links)} deck(s) on page {page}")

        for deck_url in deck_links:
            deck_id = deck_url.split("d=")[-1].split("&")[0]
            mtgo_url = get_mtgo_download_url(deck_url)
            time.sleep(REQUEST_DELAY)

            if mtgo_url:
                download_decklist(mtgo_url, deck_id)
                time.sleep(REQUEST_DELAY)
            else:
                logger.warning(f"No MTGO link found for deck {deck_id} ({deck_url})")

        current_url = get_next_page_url(soup)
        page += 1
        if current_url:
            time.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scrape_all()
