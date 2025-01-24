import os
import pdb
import tiktoken

from urllib.parse import urlparse, quote
from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import ScrapingAntLoader

load_dotenv(find_dotenv(usecwd=True))


def _scrap_pubmed_page(url: str) -> str:
    print(f"=== Load page: {url} ===")
    docs = ScrapingAntLoader(urls=[url], api_key=os.environ['SCRAPING_AND_LOADER']).load()

    return "\n\n".join([doc.page_content for doc in docs])


def load_pubmed_page(url: str) -> str:
    query = urlparse(url).query
    directory = "tmp/pubmed_fixtures"
    filename = f"{directory}/{query}.md"

    # Read from cache
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return file.read()

    # Read actual page, and cache
    page_content = _scrap_pubmed_page(url)

    os.makedirs(directory, exist_ok=True)
    with open(filename, 'w') as file:
        file.write(page_content)
    return page_content


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding('cl100k_base')
    return len(encoding.encode(text))


def load_researches(search_llm_result: list[dict]) -> list[dict]:
    result = []
    for research in search_llm_result:
        url = research['source_url']
        name = research['name']
        content = load_pubmed_page(url)

        result.append({'url': url, 'name': name, 'content': content})
    return result
