from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import List, Dict, Any
import time
import random

def build_tagger_url(set: str, collector_number: str) -> str:
    """Builds the URL for the tagger page for a given set and collector number.

    Args:
        set (str): The card set code.
        collector_number (str): The collector number of the card.

    Returns:
        str: The URL to the tagger page.
    """
    return f"https://tagger.scryfall.com/card/{set}/{collector_number}"


def scrape_tagger_page(html_content: str) -> Dict[str, List[Any]]:
    """Scrapes the tagger page HTML for card tags and relationships.

    Args:
        html_content (str): The HTML content of the tagger page.

    Returns:
        Dict[str, List[Any]]: A dictionary with 'card_tags' and 'relationships'.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Card section taggings
    card_h2 = soup.find_all('h2')[1] if len(soup.find_all('h2')) > 1 else None
    if not card_h2:
        return {'card_tags': [], 'relationships': []}
    
    card_taggings = card_h2.find_next('div', class_='taggings')
    card_tags: List[str] = []
    relationships: List[Dict[str, str]] = []
    if card_taggings:
        for tag_row in card_taggings.find_all('div', class_='tag-row'):
            icons = tag_row.find_all('div', class_=lambda x: bool(x and 'tagging-icon' in x.split()))
            if icons:
                first_icon = icons[0]
                classes = first_icon.get('class')
                last_class = classes[-1] if classes else ''
                tag_flex = tag_row.find('span', class_='tag-row-flex')
                tag_link = tag_flex.find('a') if tag_flex else None
                if tag_link:
                    tag_text: str = tag_link.get_text(strip=True)
                    if last_class != 'value-card':
                        # This is a special relationship tag
                        related_card: str = tag_text
                        relationship = last_class[6:]  # Remove 'value-' prefix
                        relationships.append({"relationship": relationship, "card": related_card})
                    else:
                        card_tags.append(tag_text)
    
        # Inherited tags (ancestors)
        ancestors_div = card_taggings.find_next('div', class_='tagging-ancestors')
        if ancestors_div:
            for a in ancestors_div.find_all('a'):
                tag: str = a.get_text(strip=True)
                card_tags.append(tag)
    
    return {"card_tags": card_tags, "relationships": relationships}

def get_tagger_data(set: str, collector_number: str) -> Dict[str, List[Any]]:
    """Retrieves tagger data for a card by scraping the tagger page.

    Args:
        set (str): The card set code.
        collector_number (str): The collector number of the card.

    Returns:
        Dict[str, List[Any]]: The scraped tagger data.
    """
    url: str = build_tagger_url(set, collector_number)

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('user-agent=BIGTRUCKPointsBot/1.0')

    driver = WebDriver(options=options)
    try:
        driver.get(url)
        # Wait for the page to load fully, e.g., wait for tagging content to appear
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.taggings'))
        )
        html_content: str = driver.page_source
    finally:
        driver.quit()

    result = scrape_tagger_page(html_content)
    time.sleep(random.uniform(0.05, 0.1)) # Sleep here because scryfall ask nicely that we don't slam their APIs; waiting for the page to render should cover us, but what's another 50-100ms among friends?
    return result