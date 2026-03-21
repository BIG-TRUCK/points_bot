from bs4 import BeautifulSoup
import requests


def build_tagger_url(oracle_id):
    return f"https://tagger.scryfall.com/search?q={oracle_id}&mode=oracle"


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


def get_tagger_data(oracle_id):
    url = build_tagger_url(oracle_id)
    response = requests.get(url)
    response.raise_for_status()
    html_content = response.text
    return scrape_tagger_page(html_content)

