"""
End-to-end pipeline:

scraping -> cleaning -> SQLite recreate/insert -> SQL queries -> output file
"""

from __future__ import annotations

from pathlib import Path

from .cleaner import GBP_TO_INR, clean_books
from .database import DEFAULT_DB_PATH, build_database, connect
from .queries import write_query_outputs
from .scraper import scrape_books


def run_pipeline() -> dict[str, object]:
    """Run every pipeline stage and print a short validation summary."""
    raw = scrape_books()
    cleaned = clean_books(raw)
    if not cleaned:
        raise RuntimeError("Cleaning produced zero rows.")

    db_path = build_database(cleaned, DEFAULT_DB_PATH)
    output_path = write_query_outputs(db_path)

    conn = connect(db_path)
    try:
        n_books = conn.execute("SELECT COUNT(*) FROM books;").fetchone()[0]
        n_cats = conn.execute("SELECT COUNT(*) FROM categories;").fetchone()[0]
        orphan = conn.execute(
            """
            SELECT COUNT(*)
            FROM books b
            LEFT JOIN categories c ON b.category_id = c.category_id
            WHERE c.category_id IS NULL;
            """
        ).fetchone()[0]
        sample = conn.execute(
            "SELECT title, price_gbp, price_inr, rating, in_stock FROM books LIMIT 3;"
        ).fetchall()
        mismatch = conn.execute(
            """
            SELECT COUNT(*) FROM books
            WHERE ABS(price_inr - (price_gbp * ?)) > 0.02;
            """,
            (GBP_TO_INR,),
        ).fetchone()[0]
    finally:
        conn.close()

    if n_books < 60:
        raise RuntimeError(f"Expected at least 60 books, found {n_books}.")
    if n_cats < 3:
        raise RuntimeError(f"Expected at least 3 categories, found {n_cats}.")
    if orphan:
        raise RuntimeError(f"Found {orphan} books with missing category foreign keys.")
    if mismatch:
        raise RuntimeError(f"Found {mismatch} rows with incorrect INR conversion.")

    print("Pipeline validation OK")
    print(f"  books={n_books}, categories={n_cats}, db={db_path}")
    print(f"  query_outputs={output_path}")
    print(f"  sample={ [dict(r) for r in sample] }")
    return {
        "n_books": n_books,
        "n_categories": n_cats,
        "db_path": str(db_path),
        "query_outputs": str(output_path),
    }


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()
