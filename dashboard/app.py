"""
SEC Financial NLP Pipeline — Streamlit Dashboard
Entry point. Run with: streamlit run dashboard/app.py

IMPORTANT: This must be run from the project root (sec-financial-nlp/),
not from inside the dashboard/ folder, since all data paths below are
relative to the project root (data/processed/...).
"""

import streamlit as st
import pandas as pd
import os
import re
from transformers import AutoTokenizer, AutoModel
from groq import Groq
from dotenv import load_dotenv
import chromadb
import torch
import tempfile
import glob
from sec_edgar_downloader import Downloader
from bs4 import BeautifulSoup


load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SEC Financial NLP Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens — palette, type, spacing (see design plan)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root {
    --ink-navy: #1A2332;
    --paper: #FAFAF7;
    --teal: #3D6B6B;
    --amber: #C08A3E;
    --border-gray: #E8E6E0;
}

h1, h2, h3 {
    font-family: 'Source Serif 4', serif;
    color: var(--ink-navy);
    font-weight: 600;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6FA8A8;
    font-weight: 500;
    margin-bottom: 0.25rem;
}

.ticker-hero {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3.5rem;
    font-weight: 700;
    color: #FAFAF7;
    line-height: 1;
    margin: 0;
}

.sentiment-bar {
    height: 6px;
    border-radius: 3px;
    margin-top: 0.75rem;
    margin-bottom: 1.5rem;
}

.limitation-box {
    background-color: #FBF3E7;
    border-left: 4px solid var(--amber);
    border-radius: 6px;
    padding: 0.85rem 1.1rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    color: #5C4520;
}
.limitation-box strong {
    color: var(--amber);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading — all functions read pre-computed CSVs, & live computation
# ---------------------------------------------------------------------------
# offline
DATA_DIR = "data/processed"

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "TSLA"]

CLUSTER_LABELS = {
    "AAPL": "Financial / Macro-Focused",
    "MSFT": "Regulatory / Disclosure-Heavy",
    "GOOGL": "Regulatory / Disclosure-Heavy",
    "JPM": "Banking / Regulatory",
    "TSLA": "Product / Manufacturing",
}

KNOWN_LIMITATIONS = {
    "MSFT": ["item1a", "anomaly_length"],
    "GOOGL": ["item1a", "anomaly_length"],
    "JPM": ["sentiment_coverage", "anomaly_length", "rag_variance"],
    "AAPL": ["rag_variance"],
    "TSLA": ["rag_variance", "sentiment_filter"],
}

LIMITATION_TEXT = {
    "item1a": "Keyword extraction for this ticker may occasionally capture forward-looking-statement disclaimer text instead of the true Risk Factors section, due to a known limitation in section-boundary detection. See the Keyword Extraction tab for details.",
    "anomaly_length": "Anomaly detection distances for this ticker are affected by filing length differences and should be interpreted within-ticker only, not compared across companies.",
    "sentiment_coverage": "Earnings-sentiment data for this ticker is limited to fewer years than the full 2020–2024 range — see the Sentiment & Trend tab for exact coverage.",
    "rag_variance": "RAG-generated answers may vary slightly between runs on borderline questions due to retrieval near-ties and model sampling. Verify against the source filing for critical use.",
    "sentiment_filter": "This ticker's earnings-related 8-K filter over-includes some non-quarterly filings; treat quarter counts as approximate.",
}


@st.cache_data
def load_csv(filename):
    """Generic cached CSV loader — avoids re-reading disk on every interaction."""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_snapshot_metrics(ticker):
    """
    Synthesizes across multiple task outputs to produce the headline
    numbers shown on the Company Snapshot tab. This is the one loader
    that combines several CSVs rather than just filtering one.
    """
    metrics = {"ticker": ticker}

    # Most recent sentiment reading
    df_sent = load_csv("sentiment_trend_results.csv")
    if df_sent is not None:
        ticker_sent = df_sent[df_sent["ticker"] == ticker].copy()
        if not ticker_sent.empty:
            ticker_sent = ticker_sent.sort_values(["year", "period_label"])
            latest = ticker_sent.iloc[-1]
            metrics["latest_sentiment_label"] = latest["label"]
            metrics["latest_sentiment_score"] = latest["score"]
        else:
            metrics["latest_sentiment_label"] = None
            metrics["latest_sentiment_score"] = None

    # Cluster assignment
    df_clusters = load_csv("cluster_results.csv")
    if df_clusters is not None:
        ticker_cluster = df_clusters[df_clusters["ticker"] == ticker]
        if not ticker_cluster.empty:
            metrics["cluster_id"] = ticker_cluster.iloc[0]["cluster"]
        else:
            metrics["cluster_id"] = None

    # Most anomalous year
    df_anomaly = load_csv("anomaly_detection_results.csv")
    if df_anomaly is not None:
        ticker_anomaly = df_anomaly[df_anomaly["ticker"] == ticker].copy()
        if not ticker_anomaly.empty:
            top = ticker_anomaly.sort_values("distance_from_centroid", ascending=False).iloc[0]
            metrics["anomalous_key"] = top["key"]
            metrics["anomalous_distance"] = top["distance_from_centroid"]

    # Most recent 10-K summary
    df_summaries = load_csv("summarization_results.csv")
    if df_summaries is not None:
        ticker_summaries = df_summaries[df_summaries["ticker"] == ticker].copy()
        if not ticker_summaries.empty:
            metrics["latest_summary"] = ticker_summaries.iloc[-1]["summary"]

    return metrics


def sentiment_color(label):
    """Maps a sentiment label to a color for the indicator bar / badges."""
    return {
        "positive": "#3D6B6B",
        "negative": "#B5533C",
        "neutral": "#A8A296",
    }.get(label, "#A8A296")


def render_limitation_box(text):
    st.markdown(f'<div class="limitation-box">⚠️ <strong>Known limitation:</strong> {text}</div>', unsafe_allow_html=True)

@st.cache_resource
def load_rag_resources():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.eval()
    chroma_client = chromadb.Client()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return tokenizer, model, chroma_client, groq_client

def embed_texts(texts, tokenizer, model, batch_size=8):
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            summed = torch.sum(token_embeddings * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            batch_embeddings = summed / counts
            all_embeddings.append(batch_embeddings)
    return torch.cat(all_embeddings, dim=0).numpy()

def chunk_filing(text, chunk_size=350, overlap=70):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

@st.cache_resource
def build_ticker_collection(ticker, _tokenizer, _model, _chroma_client):
    import pickle
    with open("data/processed/all_texts.pkl", "rb") as f:
        all_texts_fixed = pickle.load(f)

    collection_name = f"filings_{ticker}"
    try:
        _chroma_client.delete_collection(collection_name)
    except Exception:
        pass
    collection = _chroma_client.create_collection(collection_name)

    ticker_keys = [k for k in all_texts_fixed if k.startswith(f"{ticker}_10-K")]

    all_chunks, all_ids, all_metadata = [], [], []
    for key in ticker_keys:
        text = all_texts_fixed[key]
        chunks = chunk_filing(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{key}_chunk{i}")
            all_metadata.append({"filing": key, "chunk_index": i})

    embeddings = embed_texts(all_chunks, _tokenizer, _model)
    collection.add(documents=all_chunks, embeddings=embeddings.tolist(), ids=all_ids, metadatas=all_metadata)
    return collection

def retrieve_chunks(question, collection, tokenizer, model, top_k=7):
    question_embedding = embed_texts([question], tokenizer, model).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=top_k)
    return results["documents"][0], results["metadatas"][0]

def generate_answer(question, docs, groq_client, ticker):
    context = "\n\n---\n\n".join(docs)
    prompt = f"""You are a financial analyst assistant. Answer the question using ONLY the context below, from {ticker}'s SEC 10-K filings across multiple years.

Keep your answer to 2-3 sentences maximum. Be direct and specific — no preamble, no filler.
If the context contains data from multiple years, clearly state which year each figure is from — do not blend figures from different years into one answer.
If the answer is not present in the context, say "This information was not found in the provided filing excerpts" and say nothing else.

Context:
{context}

Question: {question}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=180,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def extract_clean_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    start = content.find("<DOCUMENT>")
    end = content.find("</DOCUMENT>") + len("</DOCUMENT>")
    doc = content[start:end]
    soup = BeautifulSoup(doc, "html.parser")
    for tag in soup.find_all(re.compile(r'^ix:')):
        tag.decompose()
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    first = text.find("PART I ")
    second = text.find("PART I ", first + 1)
    if second == -1:
        second = first
    if second == -1:
        second = 0
    return text[second:]


@st.cache_resource(show_spinner=False)
def fetch_live_filing(ticker):
    with tempfile.TemporaryDirectory() as tmp_dir:
        dl = Downloader("SEC-Financial-NLP-Dashboard", "3gamingworld@gmail.com", tmp_dir)
        try:
            dl.get("10-K", ticker, limit=1)
        except Exception as e:
            return None, str(e)

        files = glob.glob(f"{tmp_dir}/sec-edgar-filings/{ticker}/10-K/*/full-submission.txt")
        if not files:
            return None, "No 10-K found for this ticker."

        text = extract_clean_text(files[0])
        return text, None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 SEC Financial NLP")
    st.markdown("---")
    selected_ticker = st.selectbox("Select a company", TICKERS, key="ticker_select")
    st.markdown("---")
    st.caption(
        "An end-to-end NLP pipeline over SEC EDGAR filings — sentiment, "
        "topics, entities, keyword extraction, clustering, anomaly "
        "detection, and a retrieval-augmented Q&A assistant."
    )
    st.markdown("---")
    st.markdown("**Scope**")
    st.caption(
        "This dashboard currently covers 5 pre-loaded companies "
        "(AAPL, MSFT, GOOGL, JPM, TSLA), 2020–2024. Live EDGAR fetching "
        "and PDF upload are supported only in the **Ask This Filing** "
        "(RAG) tab — see that tab for details."
    )
    st.markdown("---")
    st.caption("Built by Chirag Jain")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_snapshot, tab_summaries, tab_sentiment, tab_topics, tab_keywords, tab_clusters, tab_anomaly, tab_rag = st.tabs(
    [
        "📋 Snapshot",
        "📄 Filing Summaries",
        "📈 Sentiment & Trend",
        "🏷️ Topics & Entities",
        "🔑 Keywords",
        "🧭 Peer Clustering",
        "🔍 Anomaly Detection",
        "💬 Ask This Filing",
    ]
)

# ---------------------------------------------------------------------------
# TAB 1 — Company Snapshot
# ---------------------------------------------------------------------------
with tab_snapshot:
    metrics = get_snapshot_metrics(selected_ticker)

    # Hero header
    label = metrics.get("latest_sentiment_label")
    bar_color = sentiment_color(label)
    st.markdown('<div class="eyebrow">Company Snapshot</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ticker-hero">{selected_ticker}</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="sentiment-bar" style="background-color:{bar_color};"></div>', unsafe_allow_html=True)

    # Known limitations for this ticker, shown once, up top
    for key in KNOWN_LIMITATIONS.get(selected_ticker, []):
        render_limitation_box(LIMITATION_TEXT[key])

    st.write("")

    # Metric cards row
    col1, col2, col3 = st.columns(3)
    with col1:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Latest Earnings Sentiment</div>', unsafe_allow_html=True)
            if label:
                st.metric(label=" ", value=label.capitalize(), delta=f"{metrics.get('latest_sentiment_score', 0):.2f} confidence")
            else:
                st.write("No data available")

    with col2:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Peer Cluster</div>', unsafe_allow_html=True)
            cluster_id = metrics.get("cluster_id")
            if cluster_id is not None:
                st.metric(label=" ", value=f"Cluster {cluster_id}")
                st.caption(CLUSTER_LABELS.get(selected_ticker, ""))
            else:
                st.write("No data available")

    with col3:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Most Atypical Filing Year</div>', unsafe_allow_html=True)
            anomalous_key = metrics.get("anomalous_key")
            if anomalous_key:
                year_match = re.search(r'-(\d{2})-\d+$', anomalous_key)
                year_full = "20" + year_match.group(1)
                st.metric(label=" ", value=year_full)
                st.caption(f"Distance score: {metrics.get('anomalous_distance', 0):.4f}")
            else:
                st.write("No data available")

    st.write("")
    st.divider()

    # Latest filing summary, boxed
    st.markdown('<div class="eyebrow">Most Recent 10-K Summary</div>', unsafe_allow_html=True)
    with st.container(border=True):
        summary = metrics.get("latest_summary")
        if summary:
            st.write(summary)
        else:
            st.write("Summary not available for this ticker.")

# tab 2
with tab_summaries:
    st.markdown('<div class="eyebrow">Filing Summaries</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ticker-hero" style="font-size: 2rem;">{selected_ticker}</p>', unsafe_allow_html=True)
    st.write("")

    df_summaries = load_csv("summarization_results.csv")

    if df_summaries is None:
        st.write("Summary data not available.")
    else:
        ticker_summaries = df_summaries[df_summaries["ticker"] == selected_ticker].copy()

        if ticker_summaries.empty:
            st.write(f"No 10-K summaries found for {selected_ticker}.")
        else:
            ticker_summaries["year"] = ticker_summaries["filing"].apply(
                lambda k: "20" + re.search(r'-(\d{2})-\d+$', k).group(1)
            )
            ticker_summaries = ticker_summaries.sort_values("year")

            for _, row in ticker_summaries.iterrows():
                st.markdown(f'<div class="eyebrow">{row["year"]} 10-K</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.write(row["summary"])
                st.write("")

with tab_sentiment:
    st.markdown('<div class="eyebrow">Sentiment & Trend</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ticker-hero" style="font-size: 2rem;">{selected_ticker}</p>', unsafe_allow_html=True)
    st.write("")

    # --- Section 1: individual sentiment readings for this ticker ---
    df_sentiment_raw = load_csv("sentiment_results.csv")
    df_sentiment_all = load_csv("sentiment_trend_results.csv")

    if df_sentiment_all is not None:
        ticker_sent = df_sentiment_all[df_sentiment_all["ticker"] == selected_ticker].copy()

        if not ticker_sent.empty:
            ticker_sent = ticker_sent.sort_values(["year", "period_label"])

            st.markdown('<div class="eyebrow">Earnings-Related 8-K Sentiment Readings</div>', unsafe_allow_html=True)
            with st.container(border=True):
                for _, row in ticker_sent.iterrows():
                    color = sentiment_color(row["label"])
                    col_a, col_b, col_c = st.columns([1, 2, 5])
                    with col_a:
                        st.markdown(f'<span style="color:{color}; font-weight:600;">{row["label"].capitalize()}</span>', unsafe_allow_html=True)
                    with col_b:
                        st.caption(f"{row['period_label']} · confidence {row['score']:.2f}")
                    with col_c:
                        quote_row = df_sentiment_raw[df_sentiment_raw["key"] == row["key"]]
                        if not quote_row.empty:
                            st.text(quote_row.iloc[0]["quote"][:120] + "...")
        else:
            st.write(f"No earnings-related sentiment data found for {selected_ticker}.")
    else:
        st.write("Sentiment trend data not available.")

    # --- Section 2: cross-company trend chart ---
    st.markdown('<div class="eyebrow">Earnings Sentiment Trend Across Companies</div>', unsafe_allow_html=True)
    render_limitation_box(
        "8-K sentiment data coverage varies by ticker (AAPL: 2022-2024, GOOGL/MSFT/TSLA: mostly 2023-2024, JPM: 2024 only) — an upstream data-pull gap, not a modeling limitation."
    )
    with st.container(border=True):
        chart_path = os.path.join(DATA_DIR, "sentiment_trend.png")
        if os.path.exists(chart_path):
            st.image(chart_path, use_container_width=True)
        else:
            st.write("Trend chart not available.")

        jpm_chart_path = os.path.join(DATA_DIR, "jpm_sentiment_2024.png")
        if os.path.exists(jpm_chart_path):
            st.image(jpm_chart_path, use_container_width=True)

with tab_topics:
    st.markdown('<div class="eyebrow">Topics, Intent & Entities</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ticker-hero" style="font-size: 2rem;">{selected_ticker}</p>', unsafe_allow_html=True)
    st.write("")

    # --- Topics: distribution chart ---
    st.markdown('<div class="eyebrow">Topic Distribution (10-K Filings)</div>', unsafe_allow_html=True)
    df_topics = load_csv("topic_results.csv")
    with st.container(border=True):
        if df_topics is not None:
            ticker_topics = df_topics[df_topics["ticker"] == selected_ticker]
            if not ticker_topics.empty:
                topic_counts = ticker_topics["topic"].value_counts()
                st.bar_chart(topic_counts)
            else:
                st.write(f"No topic data found for {selected_ticker}.")
        else:
            st.write("Topic data not available.")

    st.write("")

    # --- Intent: earnings call intent classification ---
    st.markdown('<div class="eyebrow">Intent Classification (8-K Filings)</div>', unsafe_allow_html=True)
    df_intent = load_csv("intent_results.csv")
    with st.container(border=True):
        if df_intent is not None:
            ticker_intent = df_intent[df_intent["ticker"] == selected_ticker].copy()
            if not ticker_intent.empty:
                for _, row in ticker_intent.head(10).iterrows():
                    col_a, col_b = st.columns([1, 5])
                    with col_a:
                        st.markdown(f"**{row['label'].capitalize()}**")
                        st.caption(f"{row['score']:.2f}")
                    with col_b:
                        st.text(row["quote"][:150] + "...")
                    st.write("")
                if len(ticker_intent) > 10:
                    st.caption(f"Showing 10 of {len(ticker_intent)} filings.")
            else:
                st.write(f"No intent data found for {selected_ticker}.")
        else:
            st.write("Intent data not available.")

    st.write("")

    # --- Named Entities: grouped by type, top entities ---
    st.markdown('<div class="eyebrow">Named Entities Extracted</div>', unsafe_allow_html=True)
    df_ner = load_csv("ner_results.csv")
    with st.container(border=True):
        if df_ner is not None:
            ticker_ner = df_ner[df_ner["ticker"] == selected_ticker].copy()
            if not ticker_ner.empty:
                entity_types = ticker_ner["entity_type"].unique()
                cols = st.columns(min(len(entity_types), 4))
                for i, etype in enumerate(entity_types):
                    with cols[i % len(cols)]:
                        st.markdown(f"**{etype}**")
                        top_entities = (
                            ticker_ner[ticker_ner["entity_type"] == etype]["entity_text"]
                            .value_counts()
                            .head(5)
                        )
                        for entity, count in top_entities.items():
                            st.caption(f"{entity} ({count})")
            else:
                st.write(f"No entity data found for {selected_ticker}.")
        else:
            st.write("Entity data not available.")

# Tab 5 - Keywords
with tab_keywords:
    st.markdown('<div class="eyebrow">Keyword Extraction</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ticker-hero" style="font-size: 2rem;">{selected_ticker}</p>', unsafe_allow_html=True)
    st.write("")

    if selected_ticker in ["MSFT", "GOOGL"]:
        render_limitation_box(LIMITATION_TEXT["item1a"])

    df_keywords = load_csv("keyword_results.csv")

    if df_keywords is not None:
        ticker_keywords = df_keywords[df_keywords["ticker"] == selected_ticker].copy()

        if not ticker_keywords.empty:
            ticker_keywords["year"] = ticker_keywords["filing"].apply(
                lambda k: "20" + re.search(r'-(\d{2})-\d+$', k).group(1)
            )
            years = sorted(ticker_keywords["year"].unique())

            for year in years:
                st.markdown(f'<div class="eyebrow">{year} 10-K</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    year_keywords = ticker_keywords[ticker_keywords["year"] == year].sort_values("score", ascending=False)
                    tags_html = " ".join(
                        f'<span style="background-color:#EAF0F0; color:#3D6B6B; padding:0.3rem 0.7rem; '
                        f'border-radius:14px; margin:0.2rem; display:inline-block; font-size:0.85rem;">{row["keyword"]}</span>'
                        for _, row in year_keywords.iterrows()
                    )
                    st.markdown(tags_html, unsafe_allow_html=True)
                st.write("")
        else:
            st.write(f"No keyword data found for {selected_ticker}.")
    else:
        st.write("Keyword data not available.")

# Tab 6 - Peer Clustering
with tab_clusters:
    st.markdown('<div class="eyebrow">Peer Clustering</div>', unsafe_allow_html=True)
    st.markdown('<p class="ticker-hero" style="font-size: 2rem;">All Companies</p>', unsafe_allow_html=True)
    st.caption(f"Currently viewing: **{selected_ticker}** highlighted below")
    st.write("")

    df_clusters = load_csv("cluster_results.csv")

    if df_clusters is not None:
        st.markdown('<div class="eyebrow">Cluster Assignments</div>', unsafe_allow_html=True)
        with st.container(border=True):
            for cluster_id in sorted(df_clusters["cluster"].unique()):
                cluster_tickers = df_clusters[df_clusters["cluster"] == cluster_id]["ticker"].unique()
                pills = []
                for t in cluster_tickers:
                    if t == selected_ticker:
                        pills.append(
                            f'<span style="background-color:#3D6B6B; color:white; padding:0.35rem 0.8rem; '
                            f'border-radius:14px; margin:0.2rem; display:inline-block; font-weight:600;">{t}</span>'
                        )
                    else:
                        pills.append(
                            f'<span style="background-color:#EAF0F0; color:#3D6B6B; padding:0.35rem 0.8rem; '
                            f'border-radius:14px; margin:0.2rem; display:inline-block;">{t}</span>'
                        )
                st.markdown(f"**Cluster {cluster_id}**", unsafe_allow_html=True)
                st.markdown(" ".join(pills), unsafe_allow_html=True)
                st.write("")

        st.write("")
        st.markdown('<div class="eyebrow">How Clusters Were Determined</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.write(
                "Each 10-K's Risk Factors section was embedded and grouped using K-Means clustering. "
                "The number of clusters (K=5) was chosen using the elbow method, shown below. Interestingly, "
                "this produced a perfect 1-to-1 alignment between clusters and companies — each company's 5 years "
                "of filings landed entirely within its own cluster, with zero cross-company overlap. This suggests "
                "each company's risk-factor language is distinctive and internally consistent enough that unsupervised "
                "clustering separates them by company identity alone, without needing labels."
            )
            elbow_path = os.path.join(DATA_DIR, "elbow_plot.png")
            if os.path.exists(elbow_path):
                st.image(elbow_path, use_container_width=True)
    else:
        st.write("Cluster data not available.")

# tab 7 - Anamoly detection
with tab_anomaly:
    st.markdown('<div class="eyebrow">Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="ticker-hero" style="font-size: 2rem;">{selected_ticker}</p>', unsafe_allow_html=True)
    st.write("")

    render_limitation_box(LIMITATION_TEXT["anomaly_length"])
    render_limitation_box(
        "This method is more sensitive to gradual multi-year drift than to isolated disruptive events: "
        "endpoint years (2020, 2024) are structurally more likely to register as \"most distant\" simply "
        "because they sit at the edge of any directional trend, not necessarily because something unusual "
        "occurred that year."
    )

    df_anomaly = load_csv("anomaly_detection_results.csv")

    if df_anomaly is not None:
        ticker_anomaly = df_anomaly[df_anomaly["ticker"] == selected_ticker].copy()

        if not ticker_anomaly.empty:
            ticker_anomaly["year"] = ticker_anomaly["key"].apply(
                lambda k: "20" + re.search(r'-(\d{2})-\d+$', k).group(1)
            )
            ticker_anomaly = ticker_anomaly.sort_values("year")

            st.markdown('<div class="eyebrow">Distance From Own Typical Pattern, By Year</div>', unsafe_allow_html=True)
            with st.container(border=True):
                chart_data = ticker_anomaly.set_index("year")["distance_from_centroid"]
                st.bar_chart(chart_data)

            top = ticker_anomaly.sort_values("distance_from_centroid", ascending=False).iloc[0]
            st.write("")
            with st.container(border=True):
                st.markdown(f"**Most atypical filing year: {top['year']}**")
                st.caption(f"Distance from centroid: {top['distance_from_centroid']:.4f}")
        else:
            st.write(f"No anomaly detection data found for {selected_ticker}.")
    else:
        st.write("Anomaly detection data not available.")

# Tab 8 - RAG (Retrieval-Augmented Generation)
with tab_rag:
    st.markdown('<div class="eyebrow">Ask This Filing</div>', unsafe_allow_html=True)
    st.write("")

    mode = st.radio(
        "Source",
        ["Pre-loaded Company", "Fetch Live Ticker", "Upload a Filing"],
        horizontal=True,
    )

    render_limitation_box(LIMITATION_TEXT["rag_variance"])

    tokenizer, model, chroma_client, groq_client = load_rag_resources()

    collection = None
    active_label = None

    if mode == "Pre-loaded Company":
        active_label = selected_ticker
        st.caption(f"Using pre-loaded filings for **{selected_ticker}** (2020–2024).")
        collection = build_ticker_collection(selected_ticker, tokenizer, model, chroma_client)

    elif mode == "Fetch Live Ticker":
        live_ticker = st.text_input("Enter any ticker symbol (e.g., NVDA):").strip().upper()
        if live_ticker:
            with st.spinner(f"Fetching {live_ticker}'s most recent 10-K from EDGAR — this can take a moment..."):
                text, error = fetch_live_filing(live_ticker)
            if error:
                st.write(f"Could not fetch a filing for {live_ticker}: {error}")
            else:
                active_label = live_ticker
                st.caption(f"Fetched most recent 10-K for **{live_ticker}** live from EDGAR. (Single most recent year only — not the full 5-year history available for pre-loaded companies.)")
                chunks = chunk_filing(text)
                embeddings = embed_texts(chunks, tokenizer, model)
                collection_name = f"live_{live_ticker}"
                try:
                    chroma_client.delete_collection(collection_name)
                except Exception:
                    pass
                collection = chroma_client.create_collection(collection_name)
                collection.add(
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    ids=[f"{live_ticker}_chunk{i}" for i in range(len(chunks))],
                    metadatas=[{"filing": f"{live_ticker}_live"} for _ in chunks],
                )

    elif mode == "Upload a Filing":
        uploaded_file = st.file_uploader("Upload a 10-K or 8-K as a .txt file", type=["txt"])
        if uploaded_file is not None:
            text = uploaded_file.read().decode("utf-8", errors="ignore")
            active_label = uploaded_file.name
            st.caption(f"Using uploaded file: **{uploaded_file.name}**")
            chunks = chunk_filing(text)
            embeddings = embed_texts(chunks, tokenizer, model)
            collection_name = "uploaded_filing"
            try:
                chroma_client.delete_collection(collection_name)
            except Exception:
                pass
            collection = chroma_client.create_collection(collection_name)
            collection.add(
                documents=chunks,
                embeddings=embeddings.tolist(),
                ids=[f"upload_chunk{i}" for i in range(len(chunks))],
                metadatas=[{"filing": uploaded_file.name} for _ in chunks],
            )

    st.write("")

    if collection is not None:
        question = st.text_input(f"Ask a question about {active_label}'s filing:")
        if st.button("Ask", type="primary"):
            if not question.strip():
                st.write("Please enter a question.")
            else:
                with st.spinner("Retrieving relevant excerpts and generating an answer..."):
                    docs, metas = retrieve_chunks(question, collection, tokenizer, model)
                    answer = generate_answer(question, docs, groq_client, active_label)
                    source_filings = sorted(set(m["filing"] for m in metas))

                st.write("")
                st.markdown('<div class="eyebrow">Answer</div>', unsafe_allow_html=True)
                with st.container(border=True):
                    st.text(answer)
                    st.caption(f"Sources: {', '.join(source_filings)}")
