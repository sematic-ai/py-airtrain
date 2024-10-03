import regex as re
import time

import airtrain as at
from llama_index.readers.web import RssReader

DEFAULT_URL = "https://buttondown.com/ainews/rss"

http_client = httpx.Client()

def get_urls() -> list[str]:
    """Get the URLs for the pages containing newsletter content"""
    search_page_index= 1
    pages: list[str] = []
    while search_page_index is not None:
        new_pages, search_page_index = get_archive_list_page(search_page_index)
        pages.extend(new_pages)

    # Exclude some early newsletters that were only included to test the system.
    pages = list(filter(lambda url: "newsletter-test" not in url, pages))
    return pages
    


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


def main() -> None:
    urls = get_urls()
    print(f"Will ingest {len(urls)} urls")


if __name__ == "__main__":
    main()
