import logging
import os
import time

from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join("data", "moxfield")
DECK_URL = "https://moxfield.com/decks/{deck_id}"


def download_decklist(deck_id: str) -> bool:
    """Downloads the MTGO decklist for the given Moxfield deck ID.

    Navigates to the deck page, opens the Download dialog, clicks
    'Download for MTGO', and saves the resulting .txt file to
    data/moxfield/<deck_id>.txt.

    Args:
        deck_id: The Moxfield deck ID (the path segment after /decks/).

    Returns:
        True if the file was saved successfully, False otherwise.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_dir = os.path.abspath(OUTPUT_DIR)
    dest = os.path.abspath(os.path.join(OUTPUT_DIR, f"{deck_id}.txt"))

    if os.path.exists(dest):
        logger.info(f"Already downloaded, skipping.", extra={"deck_id": deck_id, "path": dest})
        return True

    url = DECK_URL.format(deck_id=deck_id)

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=BIGTRUCKPointsBot/1.0")

    driver = WebDriver(options=options)
    wait = WebDriverWait(driver, 15)

    try:
        driver.execute_cdp_cmd("Browser.setDownloadBehavior", {
            "behavior": "allow",
            "downloadPath": download_dir,
            "eventsEnabled": True,
        })

        logger.info(f"Navigating to {url}")
        driver.get(url)
    except Exception as e:
        logger.error(f"Failed to load page for deck.", exc_info=True, extra={"deck_id": deck_id, "url": url})
        raise e
        
    try: 
        # Click the Download button to open the dialog
        download_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[.//span[normalize-space()='Download']]"))
        )
        download_btn.click()
    except Exception as e:
        logger.error(f"Failed to open download dialog for deck.", exc_info=True, extra={"deck_id": deck_id, "url": url})
        raise e

    try:
        # Click "Download for MTGO" inside the dialog
        mtgo_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//a[contains(@class,'btn-primary') and normalize-space()='Download for MTGO']"))
        )
        mtgo_btn.click()
    except Exception as e:
        logger.error(f"Failed to click MTGO download button for deck.", exc_info=True, extra={"deck_id": deck_id, "url": url})
        raise e
    
    try:
        # Wait for the file to appear in the download directory
        timeout = 15
        elapsed = 0
        downloaded = None
        while elapsed < timeout:
            time.sleep(0.5)
            elapsed += 0.5
            candidates = [
                f for f in os.listdir(download_dir)
                if f.endswith(".txt") and not f.endswith(".crdownload")
            ]
            if candidates:
                downloaded = max(
                    candidates,
                    key=lambda f: os.path.getmtime(os.path.join(download_dir, f))
                )
                break

        if not downloaded:
            logger.error(f"Download timed out for deck.", exc_info=True, extra={"deck_id": deck_id, "url": url})
            return False

        src = os.path.join(download_dir, downloaded)
        if src != dest:
            os.replace(src, dest)

        logger.info(f"Saved: {dest}")
        return True

    except Exception as e:
        logger.error(f"Failed to download deck.", exc_info=True, extra={"deck_id": deck_id, "url": url})
        return False

    finally:
        driver.quit()


def render_test(deck_id: str) -> str:
    """Navigates to a Moxfield deck page and returns the page title.

    Use this to verify that Chrome can load Moxfield before attempting downloads.
    """
    url = DECK_URL.format(deck_id=deck_id)

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=BIGTRUCKPointsBot/1.0")

    driver = WebDriver(options=options)
    try:
        logger.info(f"Navigating to {url}")
        driver.get(url)
        title = driver.title
        logger.info(f"Page title: {title}")
        return title
    finally:
        driver.quit()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 2:
        print("Usage: python decklist.py <deck_id>")
        sys.exit(1)
    download_decklist(sys.argv[1])
