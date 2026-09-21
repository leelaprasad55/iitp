"""
Reusable Books to Scrape collector.

Scrapes at least MIN_BOOKS books from at least MIN_CATEGORIES categories,
following pagination links. Nothing is hardcoded as a finished dataset.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://books.toscrape.com/"
CATALOGUE_URL = urljoin(BASE_URL, "catalogue/")

# Prefer larger categories so pagination is exercised and 60+ books are easy.
PREFERRED_CATEGORIES = [
    "Sequential Art",
    "Mystery",
    "Historical Fiction",
    "Romance",
    "Classics",
    "Travel",
    "Fiction",
    "Young Adult",
    "Science Fiction",
    "Fantasy",
]

MIN_BOOKS = 60
MIN_CATEGORIES = 3
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (compatible; ZeptoDataPipeline/1.0; student capstone scraper)"
)


def create_session() -> requests.Session:
    """Return a session with a polite User-Agent."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def fetch_page(session: requests.Session, url: str) -> str:
    """Alias required by the capstone interface: fetch one HTML page."""
    return fetch_html(session, url)


def fetch_html(session: requests.Session, url: str) -> str:
    """GET a page and return HTML text. Raises a clear error on failure."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not fetch {url}. Check your network connection. Original error: {exc}"
        ) from exc
    return response.text


def parse_category_map(home_html: str) -> dict[str, str]:
    """
    Parse the sidebar category list from the home page.

    Returns a mapping of category name -> absolute category URL.
    """
    soup = BeautifulSoup(home_html, "lxml")
    categories: dict[str, str] = {}
    side = soup.select_one("div.side_categories ul")
    if side is None:
        return categories

    for link in side.select("a"):
        name = " ".join(link.get_text().split())
        href = link.get("href")
        if not name or not href or name.lower() == "books":
            continue
        categories[name] = urljoin(BASE_URL, href)
    return categories


def parse_book(article, category_name: str, page_url: str) -> dict[str, Any]:
    """Parse one product card into a raw book record."""
    title_el = article.select_one("h3 a")
    price_el = article.select_one("p.price_color")
    rating_el = article.select_one("p.star-rating")
    avail_el = article.select_one("p.availability")

    title = title_el.get("title") if title_el else None
    if not title and title_el:
        title = title_el.get_text(strip=True)

    price = price_el.get_text(strip=True) if price_el else None

    star_rating = None
    if rating_el:
        classes = rating_el.get("class", [])
        for token in classes:
            if token != "star-rating":
                star_rating = token
                break

    availability = avail_el.get_text(" ", strip=True) if avail_el else None
    return {
        "title": title,
        "price": price,
        "star_rating": star_rating,
        "availability": availability,
        "category": category_name,
        "source_url": page_url,
    }


def parse_books_on_page(page_html: str, category_name: str, page_url: str) -> list[dict[str, Any]]:
    """Extract book cards from a single category listing page."""
    soup = BeautifulSoup(page_html, "lxml")
    books: list[dict[str, Any]] = []

    for article in soup.select("article.product_pod"):
        books.append(parse_book(article, category_name, page_url))
    return books


def next_page_url(page_html: str, current_url: str) -> str | None:
    """Return the absolute URL of the next listing page, if any."""
    soup = BeautifulSoup(page_html, "lxml")
    next_link = soup.select_one("li.next a")
    if not next_link or not next_link.get("href"):
        return None
    return urljoin(current_url, next_link["href"])


def scrape_category(
    session: requests.Session,
    category_name: str,
    start_url: str,
    remaining: int | None = None,
) -> list[dict[str, Any]]:
    """Follow pagination and collect books from one category."""
    collected: list[dict[str, Any]] = []
    url: str | None = start_url
    page_index = 1

    while url:
        html = fetch_html(session, url)
        page_books = parse_books_on_page(html, category_name, url)
        if remaining is not None:
            page_books = page_books[: max(remaining - len(collected), 0)]
        collected.extend(page_books)
        print(f"  [{category_name}] page {page_index}: +{len(page_books)} (total {len(collected)})")

        if remaining is not None and len(collected) >= remaining:
            break

        url = next_page_url(html, url)
        page_index += 1
        time.sleep(0.2)

    return collected


def choose_categories(available: dict[str, str]) -> list[str]:
    """Pick preferred categories that actually exist on the site."""
    chosen = [name for name in PREFERRED_CATEGORIES if name in available]
    if len(chosen) < MIN_CATEGORIES:
        extras = [name for name in available if name not in chosen]
        chosen.extend(extras[: MIN_CATEGORIES - len(chosen)])
    if len(chosen) < MIN_CATEGORIES:
        raise RuntimeError(
            f"Need at least {MIN_CATEGORIES} categories; found {list(available.keys())[:10]}"
        )
    return chosen


def scrape_books(
    min_books: int = MIN_BOOKS,
    min_categories: int = MIN_CATEGORIES,
) -> list[dict[str, Any]]:
    """
    Scrape books until min_books is reached from at least min_categories.

    Pagination is followed inside each category. The dataset is never hardcoded.
    """
    session = create_session()
    home_html = fetch_html(session, BASE_URL)
    category_map = parse_category_map(home_html)
    chosen = choose_categories(category_map)

    print(f"Scraping categories: {chosen}")
    all_books: list[dict[str, Any]] = []
    used_categories: list[str] = []

    for name in chosen:
        if len(all_books) >= min_books and len(used_categories) >= min_categories:
            break
        print(f"Starting category: {name}")
        books = scrape_category(session, name, category_map[name])
        all_books.extend(books)
        used_categories.append(name)

    unique_cats = {b["category"] for b in all_books}
    if len(all_books) < min_books:
        raise RuntimeError(
            f"Scraped only {len(all_books)} books; need at least {min_books}."
        )
    if len(unique_cats) < min_categories:
        raise RuntimeError(
            f"Scraped only {len(unique_cats)} categories; need at least {min_categories}."
        )

    print(f"Scraped {len(all_books)} books across {len(unique_cats)} categories.")
    return all_books


if __name__ == "__main__":
    rows = scrape_books()
    print(f"Independent run complete. Example row: {rows[0]}")
