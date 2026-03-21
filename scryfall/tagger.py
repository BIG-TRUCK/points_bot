from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def build_tagger_url(set, collector_number):
    return f"https://tagger.scryfall.com/card/{set}/{collector_number}"


def scrape_tagger_page(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the Card section taggings
    card_h2 = soup.find_all('h2')[1] if len(soup.find_all('h2')) > 1 else None
    if not card_h2:
        return {'card_tags': [], 'ancestors': []}
    
    card_taggings = card_h2.find_next('div', class_='taggings')
    card_tags = []
    relationships = []
    if card_taggings:
        for tag_row in card_taggings.find_all('div', class_='tag-row'):
            icons = tag_row.find_all('div', class_=lambda x: x and 'tagging-icon' in x.split())
            if icons:
                first_icon = icons[0]
                classes = first_icon.get('class', [])
                last_class = classes[-1] if classes else ''
                tag_link = tag_row.find('span', class_='tag-row-flex').find('a')
                if tag_link:
                    tag_text = tag_link.get_text(strip=True)
                    if last_class != 'value-card':
                        # This is a special relationship tag
                        related_card = tag_text
                        relationship = last_class[6:]  # Remove 'value-' prefix
                        relationships.append({"relationship": relationship, "card": related_card})
                    else:
                        card_tags.append(tag_text)
    
    # Inherited tags (ancestors)
    ancestors_div = card_taggings.find_next('div', class_='tagging-ancestors')
    ancestors = []
    if ancestors_div:
        for a in ancestors_div.find_all('a'):
            tag = a.get_text(strip=True)
            card_tags.append(tag)
    
    return {"card_tags": card_tags, "relationships": relationships}

def get_tagger_data(set, collector_number):
    url = build_tagger_url(set, collector_number)

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        # Wait for the page to load fully, e.g., wait for tagging content to appear
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'div.taggings'))
        )
        html_content = driver.page_source
    finally:
        driver.quit()

    return scrape_tagger_page(html_content)

