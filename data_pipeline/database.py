"""
SQLite persistence for cleaned book data.

Normalized schema:
- categories(category_id PK, category_name UNIQUE)
- books(..., category_id FK)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DB_PATH = Path(__file__).resolve().parent / "zepto_books.db"

CATEGORIES_DDL = """
CREATE TABLE categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT NOT NULL UNIQUE
);
"""

BOOKS_DDL = """
CREATE TABLE books (
    book_id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    price_gbp REAL NOT NULL,
    price_inr REAL NOT NULL,
    rating INTEGER NOT NULL,
    in_stock INTEGER NOT NULL,
    category_id INTEGER NOT NULL,
    FOREIGN KEY (category_id) REFERENCES categories(category_id)
);
"""


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open SQLite with foreign keys enabled and row factory set."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def recreate_schema(conn: sqlite3.Connection) -> None:
    """Drop and recreate tables so the pipeline can rebuild the DB from scratch."""
    conn.execute("DROP TABLE IF EXISTS books;")
    conn.execute("DROP TABLE IF EXISTS categories;")
    conn.execute(CATEGORIES_DDL)
    conn.execute(BOOKS_DDL)
    conn.commit()


def insert_cleaned_books(conn: sqlite3.Connection, cleaned_books: list[dict[str, Any]]) -> None:
    """Insert categories first, then books with matching foreign keys."""
    category_ids: dict[str, int] = {}

    for row in cleaned_books:
        name = row["category"]
        if name in category_ids:
            continue
        cursor = conn.execute(
            "INSERT INTO categories (category_name) VALUES (?);",
            (name,),
        )
        category_ids[name] = cursor.lastrowid

    for row in cleaned_books:
        conn.execute(
            """
            INSERT INTO books (
                title, price_gbp, price_inr, rating, in_stock, category_id
            ) VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                1 if row["in_stock"] else 0,
                category_ids[row["category"]],
            ),
        )
    conn.commit()


def build_database(
    cleaned_books: list[dict[str, Any]],
    db_path: str | Path = DEFAULT_DB_PATH,
) -> Path:
    """Recreate the SQLite file from cleaned rows and return the path."""
    path = Path(db_path)
    if path.exists():
        path.unlink()

    conn = connect(path)
    try:
        recreate_schema(conn)
        insert_cleaned_books(conn, cleaned_books)
        book_count = conn.execute("SELECT COUNT(*) FROM books;").fetchone()[0]
        cat_count = conn.execute("SELECT COUNT(*) FROM categories;").fetchone()[0]
        print(f"Wrote {book_count} books and {cat_count} categories to {path}")
    finally:
        conn.close()
    return path


if __name__ == "__main__":
    demo = [
        {
            "title": "Demo Book",
            "price_gbp": 10.0,
            "price_inr": 1055.0,
            "rating": 3,
            "in_stock": True,
            "category": "Travel",
        }
    ]
    build_database(demo)
