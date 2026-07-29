# SEC Financial NLP Pipeline

An end-to-end NLP pipeline over SEC EDGAR filings, deployed as an interactive Streamlit dashboard. Covers sentiment analysis, classification, NER, PII redaction, summarization, keyword extraction, clustering, retrieval-augmented Q&A, temporal trend analysis, and anomaly detection across five companies' 10-K and 8-K filings (2020–2024).

**Live demo:** https://sec-financial-nlp-chirag.streamlit.app/

---

## Overview

| | |
|---|---|
| **Companies** | AAPL, MSFT, GOOGL, JPM, TSLA |
| **Filing types** | 10-K, 8-K |
| **Date range** | 2020–2024 |
| **Total filings** | 125 |
| **Tasks** | 12 |
| **Deployment** | Streamlit Cloud |

## Features

- **Sentiment analysis** of earnings-related 8-K filings
- **Intent & topic classification** across filing content
- **Named entity recognition** (people, organizations, locations)
- **PII redaction** pipeline
- **Automated summarization** of 10-K filings
- **Risk-factor keyword extraction**
- **Peer clustering** by risk-profile similarity
- **RAG-based Q&A assistant** — ask natural-language questions about a company's filings, grounded strictly in the source text
  - Works on the 5 pre-loaded companies out of the box
  - **Fetch Live Ticker** — pull any public company's most recent 10-K on demand
  - **Upload a Filing** — analyze your own `.txt` filing
- **Temporal sentiment trend** tracking across quarters
- **Anomaly detection** — flags each company's most atypical filing year (within-company basis)

## Dashboard Tabs

1. Snapshot
2. Filing Summaries
3. Sentiment & Trend
4. Topics & Entities
5. Keywords
6. Peer Clustering
7. Anomaly Detection
8. Ask This Filing (RAG)

## Tech Stack

- **Language:** Python 3.12
- **NLP/ML:** `transformers` (embeddings via manual mean-pooling — `sentence_transformers` avoided due to a platform-specific crash), PyTorch
- **Vector store:** ChromaDB (in-memory, per-ticker collections)
- **LLM generation:** Groq (LLaMA 3.3 70B)
- **Data source:** SEC EDGAR (`sec_edgar_downloader`)
- **Dashboard:** Streamlit
- **Parsing:** BeautifulSoup

## Setup

```bash
git clone https://github.com/Chirag6667/sec-financial-nlp.git
cd sec-financial-nlp
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_key_here
```

Run the dashboard locally:

```bash
streamlit run dashboard/app.py
```

### Requirements notes

- `numpy` must stay within `>=2.0,<2.8` (compatibility constraint with `scipy`/`chromadb`).
- Do **not** install `sentence_transformers` — it is incompatible with this project's embedding approach and known to cause a crash on some Windows setups. Embeddings are generated via `transformers` directly (see `embed_texts()` in the pipeline modules).

## Project Structure

```
sec-financial-nlp/
├── dashboard/
│   ├── app.py
│   └── .streamlit/config.toml
├── notebooks/
│   ├── 01_...ipynb          # Tasks 1-4
│   ├── 04_tasks5to8.ipynb   # Tasks 5-8
│   ├── 06_tasks11to12.ipynb # Tasks 11-12
│   └── ...
├── data/
│   ├── raw/                 # Downloaded SEC filings
│   └── processed/           # Task outputs (CSVs, embeddings)
├── docs/
│   ├── TECHNICAL_REPORT.md
│   └── BUSINESS_REPORT.md
└── requirements.txt
```

## Known Limitations

This project documents its limitations directly rather than hiding them — see [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md) for full detail on each. Summary:

- **Keyword extraction (MSFT, GOOGL):** risk-factor section detection occasionally lands on a forward-looking-statements disclaimer instead of the true Risk Factors section.
- **RAG assistant:** answers on borderline questions may vary slightly between runs (near-tie retrieval + non-zero temperature — standard RAG behavior).
- **Anomaly detection:** cross-ticker distance comparisons are not valid due to a document-length confound (JPM's filings are ~2-4x longer than the others); within-ticker comparisons only. Endpoint years (2020, 2024) are structurally more likely to register as "anomalous" regardless of actual content — a known property of the leave-one-out method.
- **Sentiment trend data coverage:** varies by ticker (JPM: 2024 only; most others: 2022–2024, not full 2020–2024) due to an upstream data-pull gap.
- **Earnings-8-K filter over-inclusion (TSLA, GOOGL):** the Item 2.02 filter currently over-includes a small number of non-earnings 8-Ks for these two tickers. Confirmed and quantified; root cause not yet fully isolated. Tracked as the top-priority item for the next iteration.
- **Live-fetch/upload (RAG tab):** scoped to a single most-recent 10-K only (no multi-year history), and `.txt` uploads only (no PDF support yet). Deliberately not extended to clustering/anomaly detection/keyword extraction — see technical report for reasoning.
- **Task 10 (FinBERT fine-tuning):** deferred to a future session; ground-truth labeling strategy (self-referential vs. external dataset) still to be decided.

## Reports

- [Technical Report](docs/TECHNICAL_REPORT.md) — full methodology, debugging process, and limitations
- [Business Report](docs/BUSINESS_REPORT.md) — plain-language summary for non-technical readers

## Author

**Chirag Lalit Kumar Jain**
PG Diploma, AI & Data Science — K.J. Somaiya (9.5 CGPA)
[GitHub](https://github.com/Chirag6667) · [Kaggle](https://www.kaggle.com/chirag6668)

## License

This project is licensed under the [MIT License](https://github.com/Chirag6667/sec-financial-nlp/blob/main/LICENSE).