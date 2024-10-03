import regex as re
import time

import airtrain as at
import httpx
from bs4 import BeautifulSoup
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.string_iterable import StringIterableReader
from markdownify import markdownify

URL_TEMPLATE = "https://buttondown.com/ainews/archive/?page={page_number}"
MAX_URLS = 1000

http_client = httpx.Client()

def get_urls() -> list[str]:
    """Get the URLs for the pages containing newsletter content"""
    search_page_index= 1
    pages: list[str] = []
    while search_page_index is not None:
        new_pages, search_page_index = get_archive_list_page(search_page_index)
        pages.extend(new_pages)
        if len(pages) > MAX_URLS:
            break

    # Exclude some early newsletters that were only included to test the system.
    pages = list(filter(lambda url: "newsletter-test" not in url, pages))
    return pages[:MAX_URLS]
    


def get_archive_list_page(page_num: int) -> tuple[list[str], int | None]:
    """Get the urls for newsletter pages, along with the index of the next search page.

    When the index of the next search page is returned as None, this indicates there are no
    remaining search results.
    """
    text = http_get(URL_TEMPLATE.format(page_number=page_num))
    archive_page = BeautifulSoup(text, 'html.parser')
    link_elements = archive_page.find(class_="email-list").find_all("a")
    if link_elements is None:
        raise ValueError("Unexpected page structure")
    urls = [el["href"] for el in link_elements]
    next_index: int | None = None
    pagination = archive_page.find(class_="pagination")
    if pagination is None:
        return urls, next_index
    older_link_text = pagination.find(string=re.compile(".*Older.*"))
    if older_link_text is not None:
        next_index = page_num + 1
    return urls, next_index


def http_get(url: str) -> str:
    n_tries = 5
    while n_tries > 0:
        response = http_client.get(url)
        if response.status_code != 200:
            time.sleep(1)
        return response.text
    raise RuntimeError(
        f"Could not get URL {url}."
        "Status code: {response.status_code}. Text: {response.text}"
    )

def get_newsletter_text(url: str) -> str:
    raw_text = http_get(url)
    page = BeautifulSoup(raw_text, 'html.parser')
    content = str(page.find(class_="email-body-content"))
    prettified = markdownify(content)
    return prettified


def main() -> None:
    urls = get_urls()
    print(f"Will ingest {len(urls)} urls")
    reader = StringIterableReader()
    documents = reader.load_data(
        texts=(get_newsletter_text(url) for url in urls)
    )
    for url, document in zip(urls, documents):
        document.metadata["source"] = url
    result = at.upload_from_llama_nodes(documents, name="AI News Newsletters")
    print(f"Uploaded {result.size} rows to '{result.name}'. View at: {result.url}")

    embed_model = OpenAIEmbedding()
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Will upload {len(nodes)} newsletter chunks.")
    result = at.upload_from_llama_nodes(
        nodes,
        name="AI News Newsletter Chunks",
    )
    print(f"Uploaded {result.size} rows to {result.name}. View at: {result.url}")


if __name__ == "__main__":
    main()
