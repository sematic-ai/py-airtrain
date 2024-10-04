import regex as re
import time
from itertools import chain

import airtrain as at
import httpx
from bs4 import BeautifulSoup
from llama_index.readers.string_iterable import StringIterableReader
from llama_index.core.node_parser.file.markdown import MarkdownNodeParser
from markdownify import MarkdownConverter

URL_TEMPLATE = "https://buttondown.com/ainews/archive/?page={page_number}"
DELAY_BETWEEN_PAGES = 0.5
MAX_URLS = 1000

http_client = httpx.Client()


def get_urls() -> list[str]:
    """Get the URLs for the pages containing newsletter content"""
    search_page_index = 1
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
    archive_page = BeautifulSoup(text, "html.parser")
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
    """Perform an HTTP GET request, with retries, and return resulting raw text."""
    n_tries = 5
    sleep_interval = 1
    while n_tries > 0:
        try:
            response = http_client.get(url)
            if response.status_code != 200:
                raise RuntimeError("Bad status code")
        except Exception:
            time.sleep(sleep_interval)
            sleep_interval *= 2
        return response.text
    raise RuntimeError(
        f"Could not get URL {url}."
        "Status code: {response.status_code}. Text: {response.text}"
    )


def get_newsletter_text(url: str) -> str:
    """Get text from a newsletter page & BeautifulSoup + markdownify to clean it"""
    raw_text = http_get(url)
    page = BeautifulSoup(raw_text, "html.parser")
    content = page.find(class_="email-body-content")
    prettified = MarkdownConverter(heading_style="ATX").convert_soup(content)
    return prettified


def get_newsletter_texts(urls: list[str]) -> list[str]:
    """Get markdown text for all newsletters at the given URLs."""
    texts: list[str] = []
    for i, url in enumerate(urls):
        print(f"Getting url {i + 1}/{len(urls)}")
        texts.append(get_newsletter_text(url))
        time.sleep(DELAY_BETWEEN_PAGES)
    return texts


def main() -> None:
    start_time = time.time()
    print("Getting newsletter urls")
    urls = get_urls()
    print(f"Will ingest {len(urls)} urls")
    reader = StringIterableReader()
    newsletter_texts = get_newsletter_texts(urls)
    documents = reader.load_data(texts=newsletter_texts)
    for url, document in zip(urls, documents):
        document.metadata["source"] = url
    result = at.upload_from_llama_nodes(documents, name="AI News Newsletters")
    print(f"Uploaded {result.size} rows to '{result.name}'. View at: {result.url}")

    print("Splitting documents")
    splitter = MarkdownNodeParser()
    nodes = list(chain(*[splitter.get_nodes_from_node(doc) for doc in documents]))
    print(f"Will upload {len(nodes)} newsletter chunks.")
    result = at.upload_from_llama_nodes(
        nodes,
        name="AI News Newsletter Chunks",
    )
    print(f"Uploaded {result.size} rows to {result.name}. View at: {result.url}")
    duration = time.time() - start_time
    print(f"Completed in {duration} s")


if __name__ == "__main__":
    main()
