# Data Pipeline

This module scrapes [Books to Scrape](https://books.toscrape.com/), cleans the records, stores them in a normalized SQLite database, and runs SQL plus Pandas comparison queries.

## Run independently

From the repository root:

```bash
python -m data_pipeline.run_pipeline
```

Individual modules:

```bash
python -m data_pipeline.scraper
python -m data_pipeline.cleaner
python -m data_pipeline.database
python -m data_pipeline.queries
```

## Stages

1. `scraper.py` — live HTTP scrape with pagination (`requests` + BeautifulSoup). At least 60 books and 3 categories.
2. `cleaner.py` — `price_gbp`, integer `rating` 1–5, boolean `in_stock`, `price_inr` at **1 GBP = 105.50 INR**. Invalid numeric prices/ratings are median-imputed; rows without title/category are dropped because those identity fields cannot be imputed. Missing or unparseable availability is also dropped rather than guessed as in stock.
3. `database.py` — recreates `categories` and `books` with a foreign key.
4. `queries.py` — five SQL queries (WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, JOIN) plus Pandas `read_sql` / `merge` equivalence.
5. `run_pipeline.py` — orchestrates every stage and writes `query_outputs.txt`.

## Outputs

- `zepto_books.db`
- `query_outputs.txt`
