"""
SQL analytics plus Pandas equivalence checks.

Queries collectively demonstrate:
SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, IN/BETWEEN, and JOIN.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .database import DEFAULT_DB_PATH, connect

OUTPUT_PATH = Path(__file__).resolve().parent / "query_outputs.txt"

# Named queries used both for the text report and for pandas comparisons.
SQL_QUERIES: dict[str, str] = {
    "q1_select_where_in_stock_high_rating": """
        SELECT title, rating, price_gbp, in_stock
        FROM books
        WHERE in_stock = 1 AND rating >= 4
        ORDER BY rating DESC, title ASC;
    """,
    "q2_order_by_limit_expensive": """
        SELECT title, price_gbp, price_inr, rating
        FROM books
        ORDER BY price_gbp DESC, title ASC, book_id ASC
        LIMIT 10;
    """,
    "q3_distinct_categories": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name;
    """,
    "q4_between_mid_price": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp ASC
        LIMIT 15;
    """,
    "q5_join_books_categories": """
        SELECT b.title, b.price_gbp, b.rating, c.category_name
        FROM books AS b
        INNER JOIN categories AS c
            ON b.category_id = c.category_id
        WHERE c.category_name IN (
            SELECT category_name FROM categories ORDER BY category_name LIMIT 3
        )
        ORDER BY c.category_name, b.title
        LIMIT 20;
    """,
}


def _df_to_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "(no rows)"
    return df.to_string(index=False)


def run_sql_queries(db_path: Path = DEFAULT_DB_PATH) -> dict[str, pd.DataFrame]:
    """Execute all SQL queries and return DataFrames."""
    conn = connect(db_path)
    try:
        results = {name: pd.read_sql(sql, conn) for name, sql in SQL_QUERIES.items()}
    finally:
        conn.close()
    return results


def pandas_equivalence_checks(db_path: Path = DEFAULT_DB_PATH) -> list[str]:
    """
    Reproduce at least two SQL queries with pd.read_sql and one JOIN with pd.merge.

    Returns human-readable comparison lines. Raises if results are not equivalent.
    """
    conn = connect(db_path)
    notes: list[str] = []
    try:
        books = pd.read_sql("SELECT * FROM books;", conn)
        categories = pd.read_sql("SELECT * FROM categories;", conn)

        sql_q2 = pd.read_sql(SQL_QUERIES["q2_order_by_limit_expensive"], conn)
        pandas_q2 = (
            books.sort_values(
                ["price_gbp", "title", "book_id"],
                ascending=[False, True, True],
                kind="mergesort",
            )
            .loc[:, ["title", "price_gbp", "price_inr", "rating"]]
            .head(10)
            .reset_index(drop=True)
        )
        sql_q2_cmp = sql_q2.reset_index(drop=True)
        pd.testing.assert_frame_equal(
            sql_q2_cmp, pandas_q2, check_dtype=False, check_exact=False, atol=1e-6
        )
        notes.append(
            "EQUIVALENT: SQL q2 (ORDER BY/LIMIT) matches pandas sort_values + head(10)."
        )

        sql_q3 = pd.read_sql(SQL_QUERIES["q3_distinct_categories"], conn)
        pandas_q3 = (
            categories[["category_name"]]
            .drop_duplicates()
            .sort_values("category_name")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sql_q3.reset_index(drop=True), pandas_q3, check_dtype=False)
        notes.append("EQUIVALENT: SQL q3 (DISTINCT) matches pandas drop_duplicates.")

        sql_join = pd.read_sql(
            """
            SELECT b.title, b.price_gbp, b.rating, c.category_name
            FROM books AS b
            INNER JOIN categories AS c
                ON b.category_id = c.category_id
            ORDER BY b.book_id;
            """,
            conn,
        )
        merged = (
            pd.merge(
                books,
                categories,
                how="inner",
                on="category_id",
            )
            .loc[:, ["title", "price_gbp", "rating", "category_name"]]
            .sort_values(["title", "price_gbp", "rating", "category_name"])
            .reset_index(drop=True)
        )
        sql_join_sorted = sql_join.sort_values(
            ["title", "price_gbp", "rating", "category_name"]
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(sql_join_sorted, merged, check_dtype=False, check_exact=False)
        notes.append(
            "EQUIVALENT: SQL INNER JOIN books/categories matches pandas merge on category_id."
        )
    finally:
        conn.close()
    return notes


def write_query_outputs(
    db_path: Path = DEFAULT_DB_PATH,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Save SQL strings, result tables, and pandas comparison notes."""
    results = run_sql_queries(db_path)
    notes = pandas_equivalence_checks(db_path)

    lines: list[str] = []
    lines.append("Zepto Data Pipeline — SQL query strings and outputs")
    lines.append("=" * 72)
    lines.append("")

    for name, sql in SQL_QUERIES.items():
        lines.append(f"QUERY: {name}")
        lines.append("-" * 72)
        lines.append(sql.strip())
        lines.append("")
        lines.append("OUTPUT:")
        lines.append(_df_to_text(results[name]))
        lines.append("")
        lines.append("")

    lines.append("PANDAS vs SQL EQUIVALENCE")
    lines.append("-" * 72)
    lines.extend(notes)
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote query report to {output_path}")
    return output_path


if __name__ == "__main__":
    write_query_outputs()
