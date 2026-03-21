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
    card_h2 = soup.find('h2', string='Card')
    if not card_h2:
        return {'card_tags': [], 'ancestors': []}
    
    card_taggings = card_h2.find_next('div', class_='taggings')
    card_tags = []
    if card_taggings:
        for tag_row in card_taggings.find_all('div', class_='tag-row'):
            icons = tag_row.find_all('div', class_='tagging-icon')
            if icons:
                first_icon = icons[0]
                classes = first_icon.get('class', [])
                last_class = classes[-1] if classes else ''
                if last_class != 'value-card':
                    tag_link = tag_row.find('span', class_='tag-row-flex').find('a')
                    if tag_link:
                        tag_text = tag_link.get_text(strip=True)
                        card_tags.append({'tag': tag_text, 'class': last_class})
    
    # Ancestors
    ancestors_div = soup.find('div', class_='tagging-ancestors')
    ancestors = []
    if ancestors_div:
        for a in ancestors_div.find_all('a'):
            ancestors.append(a.get_text(strip=True))
    
    return {'card_tags': card_tags, 'ancestors': ancestors}


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

    return scrape_tagger_page(html_content), html_content

