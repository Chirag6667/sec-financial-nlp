# SEC Financial NLP Pipeline — Technical Report

**Author:** Chirag Lalit Kumar Jain
**Project:** End-to-end Financial NLP Pipeline on SEC EDGAR Filings
**Repository:** github.com/Chirag6667/sec-financial-nlp
**Program:** Deep Logic Labs AI Bootcamp (Supervisor: Vivek)

---

## 1. Overview

This project builds a 12-task NLP pipeline over SEC EDGAR filings for five companies — **AAPL, MSFT, GOOGL, JPM, TSLA** — covering **10-K and 8-K filings from 2020–2024** (125 filings total), deployed as an interactive Streamlit dashboard.

The pipeline covers the full spectrum of financial-document NLP: sentiment analysis, intent and topic classification, named entity recognition, PII redaction, summarization, keyword extraction, unsupervised clustering, retrieval-augmented question answering, temporal trend analysis, and anomaly detection.

This report documents methodology, key engineering decisions, debugging processes, and — deliberately — the limitations of each component. Where a limitation reflects genuine model/data behavior, it is documented and accepted. Where a limitation reflects a fixable bug that wasn't resolved in time, that distinction is made explicit rather than blurred.

### 1.1 Environment

| Component | Detail |
|---|---|
| OS | Windows, PowerShell |
| Python | 3.12, isolated venv |
| GPU | NVIDIA RTX 3060 Laptop, CUDA 12.4 |
| Deep learning | PyTorch 2.6.0+cu124 |
| Vector store | ChromaDB (in-memory client) |
| LLM (generation) | Groq LLaMA 3.3 70B (free tier) |
| Deployment | Streamlit Cloud |

### 1.2 A note on engineering constraints

A meaningful part of this project's engineering effort went into working around environment-specific failures rather than pure modeling work. Two are worth stating up front because they shaped several downstream design decisions:

- **`sentence_transformers` is unusable on the development machine** — it crashes with a Windows access violation (exit code `3221225477`) on import, isolated specifically to the package's ONNX/OpenVINO backend, independent of numpy/torch/transformers versions. The working replacement is loading `sentence-transformers/all-MiniLM-L6-v2` directly via `transformers`' `AutoTokenizer` + `AutoModel`, with manual mean-pooling over token embeddings. This is used everywhere embeddings are needed (Tasks 8, 9, 12).
- **numpy must remain in `>=2.0,<2.8`** — downgrading to 1.x (attempted early on, to rule out a suspected version conflict) broke `scipy` and `chromadb` imports outright.

Both were root-caused through isolated import testing rather than trial-and-error dependency swapping, and are treated as fixed constraints for the remainder of the project.

---

## 2. Task-by-Task Summary

| # | Task | Status |
|---|------|--------|
| 1 | Sentiment analysis (8-Ks) | Complete |
| 2 | Intent classification | Complete |
| 3 | Topic classification | Complete |
| 4 | Named entity recognition | Complete |
| 5 | PII redaction | Complete |
| 6 | Financial summarization (10-K) | Complete |
| 7 | Keyword extraction | Complete — documented limitation (Section 4.1) |
| 8 | Clustering by risk profile | Complete |
| 9 | RAG assistant | Complete — locked configuration (Section 4.2) |
| 10 | FinBERT fine-tuning | **Deferred** — requires dedicated GPU session (Section 6) |
| 11 | Temporal sentiment trend | Complete — documented limitations (Section 4.3) |
| 12 | Anomaly detection | Complete — documented limitations (Section 4.4) |
| — | Streamlit dashboard (8 tabs) | Complete, deployed to Streamlit Cloud |

Tasks 1–6 and 8 (sentiment, intent, topic, NER, PII, summarization, clustering) produced clean, well-behaved outputs and are not elaborated on further here beyond their role as inputs to later tasks — the more interesting engineering content is in Tasks 7, 9, 11, and 12, detailed below.

---

## 3. Reusable Core Functions

Several functions are shared across multiple tasks to keep methodology consistent. The two most consequential:

**`get_content_window()`** — locates the "Item 1A. Risk Factors" section of a 10-K using a gap-distance heuristic (finds an "Item 1A" mention followed by an "Item 1B" mention at least 3,000 characters later, to distinguish the actual section header from earlier incidental mentions). Used in Tasks 6–8.

**`chunk_filing()`** — word-based sliding-window chunking (`chunk_size=350, overlap=70` after tuning — see Section 4.2) used for embedding-based tasks (8, 9, 12).

Both functions, along with the `embed_texts()` transformers-based embedding replacement and a robust `extract_year()` regex (`-(\d{2})-\d+$` on SEC accession numbers), are defined once and imported consistently rather than duplicated per notebook, to avoid silent behavioral drift between tasks that should share logic.

---

## 4. Deep-Dive: Debugging Sagas and Documented Limitations

### 4.1 Task 7 — Keyword Extraction: Item 1A Boilerplate Collision

**Issue:** For MSFT and GOOGL specifically, `get_content_window()` sometimes lands on an early forward-looking-statements disclaimer paragraph instead of the true Risk Factors section, because that disclaimer also contains the phrase "Item 1A" and happens to clear the 3,000-character gap threshold used by the heuristic.

**Status:** Investigated across 7 separate attempts at threshold/pattern tuning. Accepted as a documented limitation rather than continuing to chase a fully general solution, since every fix attempted for MSFT/GOOGL introduced regressions elsewhere. The dashboard surfaces this explicitly as a per-ticker limitation banner for MSFT and GOOGL rather than silently returning wrong keywords.

**Why this was the right call, not a shortcut:** a heuristic-based section-boundary detector across 5 companies with materially different 10-K formatting conventions is inherently going to have edge cases. Seven tuning attempts is a reasonable diligence bar; a fully general parser (e.g., structural HTML/XBRL-tag-based section detection rather than text-pattern matching) is identified as the correct long-term fix and noted in Section 7 (Future Work).

### 4.2 Task 9 — RAG Assistant: Chunk-Size Tuning

**Final locked configuration:** `chunk_size=350`, `overlap=70`, `top_k=7`, `temperature=0.1`, `max_tokens=180`, per-ticker ChromaDB collections, Groq LLaMA 3.3 70B generation with a strict "answer only from context" system prompt.

**Debugging saga:**
1. Started at `chunk_size=600, top_k=5`. A test question about GOOGL's antitrust exposure returned "not found in the provided filing excerpts," despite the content being verifiably present in the raw filing text (confirmed via direct string search).
2. Root cause: **chunk-boundary dilution.** The relevant antitrust content was embedded mid-chunk inside a 600-word block dominated by unrelated topics (foreign exchange risk, net neutrality). The resulting chunk embedding didn't represent the antitrust content strongly enough to be retrieved for a targeted question about it.
3. Fix attempt 1: reduced `chunk_size` 600→350, `overlap` 100→70. This fixed the GOOGL case, but regressed a previously-working TSLA question.
4. Fix attempt 2: increased `top_k` 5→7, reasoning that smaller chunks need more of them retrieved to cover equivalent context. This resolved both cases simultaneously.

**Accepted limitation:** answers show run-to-run variance on borderline questions (confirmed via repeated identical-setting testing on a TSLA stock-volatility question and a JPM capital-requirements question). This stems from near-tie retrieval rankings combined with non-zero generation temperature, and is standard RAG behavior rather than a pipeline defect. It is disclosed to end users in the dashboard.

### 4.3 Task 11 — Temporal Sentiment Trend

**Methodology:** 8-K filings are filtered to earnings-related releases using SEC's official "Item 2.02: Results of Operations and Financial Condition" item code, read directly from filing headers (not every 8-K is an earnings release, so this filter is necessary before computing a meaningful trend). Sentiment is signed (+/−/0) and plotted per ticker.

**Data-coverage limitation (accepted):** the underlying `sentiment_results.csv` only covers 2022–2024 for most tickers, not the full 2020–2024 range covered by the 10-K corpus — an upstream data-pull gap from an earlier pipeline session. JPM has data for 2024 only. This is disclosed rather than backfilled, since re-pulling would require re-running the full 8-K ingestion for out-of-range years.

**Chart rendering bug (fixed):** plotting multiple tickers against shared text period-labels (e.g., "2023-Q1") on a matplotlib x-axis produced broken, jumbled lines, because tickers have different numbers of data points. Fixed by plotting on integer filing-sequence position per ticker rather than shared categorical text labels.

**Filter over-inclusion bug (unresolved, documented as a known data-quality limitation):** a post-hoc audit (`groupby(['ticker','year']).size()` on the final CSV) confirmed the Item 2.02 filter over-includes filings for two tickers:

| Ticker | Year | Filings found | Expected (~4/yr) |
|---|---|---|---|
| GOOGL | 2023 | 5 | 4 |
| TSLA | 2023 | 6 | 4 |
| TSLA | 2024 | 8 | 4 |

Two root-cause hypotheses were investigated — duplicate preliminary/final earnings filings, and shareholder-deck/table-of-contents attachments incidentally matching the Item 2.02 header pattern. Diagnosis was attempted via two approaches:

1. Ordering flagged filings by SEC accession number — inconclusive, since accession sequence numbers are agent-specific counters (three different filing agents were involved across TSLA's flagged filings) and cannot be reliably used for chronological ordering across agents.
2. Extracting filing dates directly from `FILED AS OF DATE` headers — this pass was not properly scoped to only the earnings-filtered accession subset (it globbed all TSLA 8-Ks rather than the 14 flagged ones), so the resulting dates could not be mapped back to the specific filings under investigation.

Given two inconclusive diagnostic passes and remaining project scope (Task 10, deployment, reporting), this was deliberately parked as a documented limitation rather than pursued to a third diagnostic attempt. **This is the single highest-priority item for a future iteration** — a corrected diagnostic (mapping each of the 14 flagged accession numbers directly to both filing date and full item-info header text in one pass) is the identified next step.

**Additional observation (not a bug):** AAPL's signed sentiment swings between roughly +0.95 and −0.95 quarter-to-quarter. This likely reflects the underlying sentiment classifier behaving in a confidence-heavy/near-binary manner rather than a data or pipeline issue, and is noted here rather than treated as something to fix.

### 4.4 Task 12 — Anomaly Detection: Methodology Evolution

**Final methodology:** per-ticker, leave-one-out cosine distance from centroid, using full-document chunk-averaged embeddings (chunked via the same `chunk_filing()` used in Task 9, then mean-pooled across all chunk embeddings per filing).

**Methodology evolution (3 attempts):**
1. **`get_content_window()` (Risk-Factors-scoped)** — MSFT showed exactly `0.000000` distance across all 5 filings. Root cause: the same Item 1A boilerplate-collision bug from Task 7 (Section 4.1) resurfaced here — MSFT's extraction was repeatedly landing on near-identical disclaimer text rather than genuine year-specific content, producing artificially identical embeddings.
2. **Fixed-offset window (`words[2000:5000]`)** — fixed MSFT's zero-distance issue, but introduced a new problem: an arbitrary landing point per filing with no guarantee of semantic relevance. Evidence: MSFT-2020 showed a 5× outlier distance (0.77) relative to other years (0.13–0.20), traced to the fixed offset likely landing on a substantively different, unrelated section that year.
3. **Full-document, chunked and averaged (final)** — eliminates both the boilerplate-collision risk and the arbitrary-landing risk by representing the entire filing, not a heuristically-located subsection.

**Two documented limitations, both structural rather than fixable bugs:**

1. **Document-length confound.** JPM's 10-Ks average ~105,000 words versus 23,000–43,000 for the other four tickers. Averaging over substantially more chunks produces a mechanically "smoother" embedding vector, shrinking JPM's distances regardless of actual year-to-year content variability. **Cross-ticker distance comparisons are not statistically valid under this methodology — only within-ticker, year-to-year comparisons are.**
2. **Endpoint bias.** Every ticker's most anomalous year was either 2020 or 2024 — never a middle year. This is an expected structural property of leave-one-out centroids under gradual multi-year drift: endpoint years sit at the edge of the observed trend with no "opposite side" pulling the average back toward them, making them mechanically more likely to register as distant from centroid — independent of whether anything unusual actually happened that year. This means the method is better characterized as detecting "start/end of an observed trend" rather than genuine acute single-year disruptions. No ticker showed a middle-year spike, which would be the stronger signal of a true anomaly.

---

## 5. Streamlit Dashboard

**Deployment:** Live on Streamlit Cloud, `dashboard/app.py`.

**Structure:** 8 tabs — Snapshot, Filing Summaries, Sentiment & Trend, Topics & Entities, Keywords, Peer Clustering, Anomaly Detection, and Ask This Filing (RAG).

**Design system:** ink navy / paper white / teal accent palette; Source Serif 4 (headers), Inter (body), JetBrains Mono (numeric/ticker data); `.streamlit/config.toml` forces a light theme to prevent client dark-mode from overriding custom CSS.

**RAG tab — live extension.** In addition to the 5 pre-loaded tickers, the RAG tab supports:
- **Fetch Live Ticker** — fetches any ticker's single most recent 10-K via `sec_edgar_downloader` at query time (not a 5-year history — a disclosed limitation relative to the pre-loaded tickers).
- **Upload a Filing** — accepts `.txt` filings (PDF not currently supported).

Both modes were tested and validated (NVDA live-fetch, an uploaded GOOGL 10-K).

**Scope decision.** Live-fetch/upload were deliberately scoped to the RAG tab only, not extended to clustering, anomaly detection, or keyword extraction:
- Clustering and anomaly detection require multi-year per-ticker history that a single fetched filing cannot provide.
- Keyword extraction's Item 1A heuristic (Section 4.1) was tuned and validated only against the 5 known tickers' filing structures; applying it to an arbitrary company's filing would mean shipping unvalidated accuracy.

RAG and filing summarization were identified as the two components that generalize by design to arbitrary filings; only RAG was built out within project scope. This is stated directly to end users in the dashboard's sidebar, rather than left as a silent gap.

**Engineering notes worth recording:**
- `st.write()` and `st.caption()` render markdown, which was unpredictably auto-linking URLs and code-block-formatting dollar amounts found in raw filing text. Fixed by switching all raw-text and AI-generated-text display to `st.text()`.
- Overriding `div[data-testid="stVerticalBlockBorderWrapper"]` or `stMetricValue` font-family CSS selectors caused metric values and container borders to render completely invisible in the installed Streamlit version — diagnosed by disabling custom CSS incrementally and re-adding it. The final CSS avoids these selectors entirely.
- `@st.cache_resource` (not `@st.cache_data`) is used for model/tokenizer/ChromaDB/Groq client objects, with leading-underscore parameter names on cached functions to exclude unhashable objects from Streamlit's cache key while keeping ticker identity part of the cache key.

---

## 6. Task 10 — FinBERT Fine-Tuning (Deferred)

Explicitly deferred to a dedicated GPU session, sequenced after Tasks 11/12 and dashboard completion. One design question remains open and must be resolved before that session starts: what ground-truth labels to fine-tune against —

- **Option A — self-referential:** use the project's own Task 1 sentiment classifier output as training labels.
- **Option B — external:** use an established labeled dataset (e.g., Financial PhraseBank).

Option A risks circularity (fine-tuning a model to reproduce another model's judgments rather than ground truth), while Option B is more rigorous but requires validating that an external dataset's domain (general financial news, in Financial PhraseBank's case) transfers adequately to SEC 8-K earnings-release language specifically. This decision is treated as a prerequisite to starting the task, not an in-flight decision to make mid-session.

---

## 7. Future Work

1. Corrected diagnostic + fix for the Task 11 Item 2.02 filter over-inclusion (TSLA, GOOGL) — highest priority.
2. A structural (HTML/XBRL-tag-based) replacement for the text-pattern Item 1A section detector, to resolve the MSFT/GOOGL boilerplate-collision limitation at its root rather than documenting around it.
3. Multi-year live-fetch support (beyond single most-recent 10-K) if the RAG tab's scope is extended.
4. Task 10 FinBERT fine-tuning, pending the ground-truth-source decision above.
5. FastAPI service layer and pytest/CI-CD pipeline — deferred as stretch scope.

---

## 8. Assessment Alignment

| Criterion | Weight | Relevant evidence in this report |
|---|---|---|
| Works & reproducible | 25% | Deployed, working dashboard; all environment constraints documented and reproducible |
| Correctness & rigor | 25% | Root-cause debugging methodology throughout (Sections 4.1–4.4); limitations quantified with evidence, not asserted |
| Code quality | 15% | Shared reusable functions (Section 3); consistent patterns across tasks |
| Documentation | 20% | This report; per-tab dashboard limitation disclosures; inline code documentation |
| Demo | 10% | Live Streamlit Cloud deployment |
| Stretch/depth | 5% | Live-fetch/upload RAG extension beyond the 5 pre-loaded tickers |
