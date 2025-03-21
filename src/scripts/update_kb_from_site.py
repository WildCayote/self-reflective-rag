import json
import os
from urllib.parse import urljoin
from dotenv import load_dotenv, find_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from embedding_service import PineconeEmbeddingManager
import schedule
import time
import requests
from bs4 import BeautifulSoup

def update_knowledge_base():

    load_dotenv(find_dotenv())
    api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_index = os.environ.get('INDEX_NAME')
    pinecone_namespace = os.environ.get('NAMESPACE')

    manager = PineconeEmbeddingManager(api_key=api_key, index_name=pinecone_index, name_space=pinecone_namespace)
  
    manager.create_and_store_embeddings()

def get_all_links(url):
    """Extract all internal links from the given URL."""
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    links = set()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        full_url = urljoin(BASE_URL, href)  # Convert to absolute URL

        # Ensure we only get internal links
        if BASE_URL in full_url and full_url not in visited_urls:
            links.add(full_url)

    return links

def scrape_page(url):
    """Scrape text content from a given URL."""
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.text.strip() if soup.title else "No Title"
    paragraphs = [p.text.strip() for p in soup.find_all("p") if p.text.strip()]

    return {"url": url, "title": title, "content": "\n".join(paragraphs)}

def scrape_website():
    """Crawl the website and extract text from all pages."""
    pages_to_visit = get_all_links(BASE_URL)  # Start with homepage links

    while pages_to_visit:
        url = pages_to_visit.pop()
        if url in visited_urls:
            continue

        print(f"Scraping: {url}")
        try:
            page_data = scrape_page(url)
            data.append(page_data)
            visited_urls.add(url)
            time.sleep(1)  # Be respectful and avoid rapid requests

            # Discover new links from this page
            new_links = get_all_links(url)
            pages_to_visit.update(new_links - visited_urls)

        except Exception as e:
            print(f"Failed to scrape {url}: {e}")

    # Save to JSON file
    with open("kifiya_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print("Scraping complete. Data saved to kifiya_data.json")

# Run the scraper

if __name__ == '__main__':

    BASE_URL = os.environ.get('BASE_URL_TO_SCRAPE')
    headers = os.environ.get('HEADERS')
    visited_urls = set()
    data = []

    scrape_website()
    # update_knowledge_base()  
        
    # Schedule the update_knowledge_base function to run every 2 weeks
    schedule.every(2).weeks.do(update_knowledge_base)

    while True:
        schedule.run_pending()
        time.sleep(1)