"""
Clean raw Books-to-Scrape records into analysis-ready columns.

Conversions:
- price text -> numeric price_gbp
- star rating word -> integer 1-5
- availability text -> boolean in_stock
- price_inr using a fixed rate of 1 GBP = 105.50 INR (no live FX API)

Missing / invalid numeric values use median imputation. Rows without a title or
category are dropped because those keys cannot be imputed meaningfully.
"""

from __future__ import annotations

import statistics
import re
from typing import Any

GBP_TO_INR = 105.50

RATING_MAP = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}

PRICE_PATTERN = re.compile(r"[\d.]+")


def parse_price_gbp(raw_price: Any) -> float | None:
    """Extract a numeric GBP price from strings like '£51.77'."""
    if raw_price is None:
        return None
    text = str(raw_price).replace(",", "").strip()
    match = PRICE_PATTERN.search(text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def parse_rating(raw_rating: Any) -> int | None:
    """Convert One/Two/.../Five (or digits) into an integer 1-5."""
    if raw_rating is None:
        return None
    text = str(raw_rating).strip().lower()
    if text.isdigit():
        value = int(text)
        return value if 1 <= value <= 5 else None
    return RATING_MAP.get(text)


def parse_in_stock(raw_availability: Any) -> bool | None:
    """Treat 'In stock' style text as True and 'Out of stock' as False."""
    if raw_availability is None:
        return None
    text = str(raw_availability).strip().lower()
    if "out of stock" in text:
        return False
    if "in stock" in text:
        return True
    return None


def _median_or_default(values: list[float], default: float) -> float:
    if not values:
        return default
    return float(statistics.median(values))


def clean_books(raw_books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Return cleaned rows.

    Output keys: title, price_gbp, price_inr, rating, in_stock, category
    """
    parsed: list[dict[str, Any]] = []
    dropped_identity = 0

    for row in raw_books:
        title = (row.get("title") or "").strip()
        category = (row.get("category") or "").strip()
        if not title or not category:
            dropped_identity += 1
            continue
        parsed.append(
            {
                "title": title,
                "category": category,
                "price_gbp": parse_price_gbp(row.get("price")),
                "rating": parse_rating(row.get("star_rating")),
                "in_stock": parse_in_stock(row.get("availability")),
            }
        )

    valid_prices = [r["price_gbp"] for r in parsed if r["price_gbp"] is not None and r["price_gbp"] >= 0]
    valid_ratings = [r["rating"] for r in parsed if r["rating"] is not None]
    median_price = _median_or_default(valid_prices, 0.0)
    median_rating = int(round(_median_or_default(valid_ratings, 3.0)))
    median_rating = min(5, max(1, median_rating))

    cleaned: list[dict[str, Any]] = []
    imputed_price = 0
    imputed_rating = 0
    imputed_stock = 0

    for row in parsed:
        price_gbp = row["price_gbp"]
        if price_gbp is None or price_gbp < 0:
            price_gbp = median_price
            imputed_price += 1

        rating = row["rating"]
        if rating is None:
            rating = median_rating
            imputed_rating += 1

        in_stock = row["in_stock"]
        if in_stock is None:
            # Boolean fields are not numeric; assume in stock because that is
            # the dominant listing state on Books to Scrape, then keep the row.
            in_stock = True
            imputed_stock += 1

        cleaned.append(
            {
                "title": row["title"],
                "price_gbp": round(float(price_gbp), 2),
                "price_inr": round(float(price_gbp) * GBP_TO_INR, 2),
                "rating": int(rating),
                "in_stock": bool(in_stock),
                "category": row["category"],
            }
        )

    print(
        f"Cleaned {len(cleaned)} books; dropped {dropped_identity} rows without title/category; "
        f"imputed price={imputed_price}, rating={imputed_rating}, in_stock={imputed_stock}."
    )
    return cleaned


if __name__ == "__main__":
    sample = [
        {
            "title": "Demo Book",
            "price": "£10.00",
            "star_rating": "Three",
            "availability": "In stock",
            "category": "Travel",
        },
        {
            "title": "Broken Price",
            "price": "unknown",
            "star_rating": "Four",
            "availability": "In stock",
            "category": "Travel",
        },
    ]
    print(clean_books(sample))
