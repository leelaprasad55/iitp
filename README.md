# Zepto Data & AI Platform

## 1. Project title

Zepto Data & AI Platform — a three-module capstone covering a Books-to-Scrape data pipeline, Titanic analytics and machine learning, and a local RAG support assistant.

## 2. Project overview

This repository is a complete, locally runnable grade-oriented implementation. Module 1 scrapes, cleans, and stores book data in SQLite, then compares SQL with Pandas. Module 2 profiles the Titanic dataset, trains classifiers and a fare regressor, and saves a reloadable sklearn pipeline. Module 3 answers Zepto-style policy questions with ChromaDB retrieval and LangGraph routing, exposed through FastAPI and Docker. Paid cloud APIs are optional and never required.

## 3. Problem statement

A quick-commerce company needs (1) a reproducible product catalog pipeline, (2) transparent survival and fare models on a public passenger dataset used as a stand-in for tabular operations analytics, and (3) a support assistant that can answer policy questions from a fixed document corpus without calling a paid LLM.

## 4. Architecture

```
Books to Scrape  →  scraper → cleaner → SQLite  →  SQL + Pandas
seaborn Titanic  →  titanic.csv  →  01_eda.ipynb / 02_modeling.ipynb  →  joblib
policy docs      →  chunk → MiniLM embed → ChromaDB → LangGraph → FastAPI :7860
```

## 5. Technologies used

Python, requests, BeautifulSoup, pandas, SQLite, seaborn, matplotlib, scikit-learn, imbalanced-learn, joblib, Jupyter, sentence-transformers (`all-MiniLM-L6-v2`), ChromaDB, LangGraph, Pydantic, FastAPI, uvicorn, Docker.

## 6. Repository structure

```
zepto-data-ai-platform/
├── README.md
├── .gitignore
├── requirements.txt
├── data_pipeline/
├── analytics/
└── support_assistant/
```

## 7. Data Pipeline explanation

`data_pipeline/scraper.py` live-scrapes at least 60 books from at least 3 Books-to-Scrape categories and follows pagination. `cleaner.py` builds `price_gbp`, integer `rating`, boolean `in_stock`, and `price_inr` at **1 GBP = 105.50 INR**. `database.py` recreates normalized `categories` and `books` tables with a foreign key. `queries.py` runs five SQL queries (WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN/IN, JOIN) and proves Pandas `read_sql` / `merge` equivalence. `run_pipeline.py` orchestrates the full path and writes `query_outputs.txt`.

## 8. Analytics explanation

`01_eda.ipynb` loads Titanic **once** via `sns.load_dataset("titanic")`, saves `analytics/titanic.csv`, then profiles missingness with the specified drop/impute/encode rules. It includes univariate age/fare charts and IQR outliers, bivariate survival with boolean masks, a six-column correlation heatmap (no `adult_male` / `alone`), four multivariate charts with written interpretation, and a z-score demonstration that is **not** leaked into modeling.

## 9. Machine Learning explanation

`02_modeling.ipynb` reads only `titanic.csv`. A stratified split happens first. A `ColumnTransformer` pipeline imputes and one-hot encodes, fitted on train only. Logistic Regression, Decision Tree, and Random Forest share that split and are scored with confusion matrices, accuracy, precision, recall, F1, ROC, and ROC-AUC. Imbalance compares baseline vs `class_weight='balanced'` vs **SMOTE on training data only**. GridSearchCV tunes Random Forest; a bootstrap forest reports **OOB score**. A separate model predicts `fare` (MAE, RMSE, R², adjusted R², residual plot). The winning classifier is saved as `analytics/models/best_pipeline.joblib` and reloaded on raw rows.

## 10. RAG Support Assistant explanation

Eight educational policy documents are chunked, embedded with MiniLM, and stored in ChromaDB. LangGraph nodes `classify_intent`, `retrieve_and_answer`, and `direct_answer` route policy keywords to top-3 retrieval. MOCK_LLM answers are deterministic: retrieval answers start with `Based on the retrieved context:` and general questions receive `I can only answer questions about Zepto policies right now.`

## 11. Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -r support_assistant/requirements.txt
```

## 12. Environment setup

Copy `.env.example` if you use an optional LLM. Default grading path needs **no** API keys.

| Variable | Meaning |
| --- | --- |
| `MOCK_LLM` | Unset or `1` = local mock (default). `0` = optional real LLM |
| `OPENAI_API_KEY` | Only for `MOCK_LLM=0` |
| `OPENAI_BASE_URL` | Optional OpenAI-compatible base URL |
| `OPENAI_MODEL` | Optional model name |

## 13. How to run Data Pipeline

From the repository root:

```bash
python -m data_pipeline.run_pipeline
```

## 14. How to run Analytics notebooks

```bash
jupyter notebook analytics/01_eda.ipynb
jupyter notebook analytics/02_modeling.ipynb
```

Non-interactive:

```bash
python analytics/_execute_notebook.py analytics/01_eda.ipynb
python analytics/_execute_notebook.py analytics/02_modeling.ipynb
```

Run `01_eda.ipynb` first so `titanic.csv` exists.

## 15. How to run Support Assistant

```bash
python -m support_assistant.embeddings
python -m support_assistant.graph
python -m support_assistant.main
```

## 16. FastAPI usage

```bash
uvicorn support_assistant.main:app --host 0.0.0.0 --port 7860
```

`POST /ask` with `{"query": "..."}`. `GET /health` returns `{"status": "ok"}`.

## 17. Docker usage

```bash
cd support_assistant
docker build -t zepto-support .
docker run --rm -p 7860:7860 -e MOCK_LLM=1 zepto-support
```

Hugging Face Spaces deployment is optional and not required for grading.

## 18. MOCK_LLM explanation

If `MOCK_LLM` is unset or `1`, intent uses a lowercase keyword heuristic (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`). Retrieval still embeds the query and reads the top 3 Chroma chunks. Generation does **not** call an external LLM. `MOCK_LLM=0` is optional and reads keys from the environment only.

## 19. Example API requests/responses

Captured from a live local `/ask` call with `MOCK_LLM=1` on 2026-09-21.

**Policy query**

```http
POST /ask
{"query": "What is the return policy?"}
```

```json
{"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unop","sources":["doc_02.txt","doc_06.txt","doc_05.txt"],"confidence":1.0}
```

**General query**

```http
POST /ask
{"query": "Tell me a joke"}
```

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## 20. Testing instructions

```bash
python -m data_pipeline.run_pipeline
python -m support_assistant.ingest
python -m support_assistant.embeddings
python -m support_assistant.graph
```

Check: ≥60 books, ≥3 categories, `zepto_books.db` foreign keys, `query_outputs.txt` pandas equivalence, `analytics/titanic.csv`, charts under `analytics/outputs/charts/`, metrics under `analytics/outputs/metrics/`, `analytics/models/best_pipeline.joblib`, FastAPI `/ask`, Docker build.

## 21. Git workflow

Work starts on `main`. Feature work happens on a branch (for example `feature/data-pipeline`) with at least two meaningful commits, then the branch is merged back into `main`. History is real (`git log --graph --all --oneline`); it is not rewritten.

## 22. Limitations

Books-to-Scrape must be reachable. Titanic modeling uses a small public dataset, not live Zepto orders. Fare regression can show heteroscedasticity on luxury tickets. RAG answers only the eight sample documents. MOCK_LLM answers are extractive, not conversational. Docker images download the MiniLM weights on first run.

## 23. Future improvements

Scheduled scrape jobs, a real product catalog instead of books, calibration of classifier thresholds for operations, hybrid search, evaluation sets for RAG, and an optional free local LLM (for example Ollama) behind `MOCK_LLM=0`.
