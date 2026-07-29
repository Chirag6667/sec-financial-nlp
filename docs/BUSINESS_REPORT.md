# SEC Financial NLP Pipeline — Business Summary

**Prepared by:** Chirag Lalit Kumar Jain
**Project:** Automated Analysis Tool for Public Company Financial Filings

---

## What This Is

Public companies are legally required to file regular reports with the U.S. Securities and Exchange Commission (SEC) — annual reports (10-Ks) and event disclosures (8-Ks). These filings contain valuable information for investors and analysts, but they're long, dense, and time-consuming to read manually. A single 10-K can run over 100 pages.

This project builds a tool that reads these filings automatically and surfaces the information an analyst would otherwise have to dig for by hand — sentiment, risk factors, key topics, notable people and organizations mentioned, year-over-year trends, and unusual filings that deviate from a company's normal pattern. It covers five well-known companies — **Apple, Microsoft, Google (Alphabet), JPMorgan Chase, and Tesla** — across five years of filings (2020–2024), and is delivered as an interactive dashboard anyone can explore without needing to read the raw filings themselves.

## What It Does

The dashboard is organized into eight views, covering:

- **At-a-glance snapshot** — the latest sentiment reading, which peer group a company's risk profile most resembles, and its most unusual filing year, all in one place.
- **Filing summaries** — plain-language summaries of each year's annual report.
- **Sentiment trends over time** — whether a company's tone in its quarterly earnings announcements has been getting more positive or negative, tracked year over year.
- **Key topics and named entities** — what subjects each filing focuses on, and which people, organizations, and locations it mentions most.
- **Risk-factor keywords** — the terms that dominate each company's stated risk factors.
- **Peer grouping** — which companies' risk profiles most resemble each other, based on the actual language in their filings (not just industry labels).
- **Anomaly detection** — flags which year, for each company, reads most differently from that company's own typical filing pattern.
- **Ask a question** — a chat-style assistant that answers plain-English questions about a company's filings, citing only what's actually in the source documents. This also works on companies outside the original five — a user can fetch any public company's most recent annual report on demand, or upload their own filing, and ask questions about it directly.

## Where This Adds Value

- **Time savings.** Reading and cross-referencing five years of filings for one company manually can take hours; the dashboard surfaces the same information in seconds.
- **Consistency.** The same analytical lens (sentiment scoring, topic detection, risk-keyword extraction) is applied identically across every company and year, removing the inconsistency that comes from manual, subjective reading.
- **Extensibility.** The question-answering feature isn't limited to the five pre-loaded companies — it can pull and analyze any public company's latest filing on request.

## Known Limitations (Stated Plainly)

No automated system reading dense financial text is perfect, and being upfront about where this one falls short is part of using it responsibly:

- **A handful of companies' risk-factor keyword extraction occasionally pulls from the wrong section of the filing** (a generic legal disclaimer instead of the actual risk factors), affecting Microsoft and Google specifically. This is flagged directly in the dashboard wherever it applies.
- **The question-answering assistant can occasionally give slightly different wording between two runs of the same borderline question** — a known characteristic of this type of AI system, not a data error. Answers are always grounded in the actual filing text, never invented.
- **The "unusual year" detection tool works well for spotting a company's outlier year relative to its own history, but its results should not be used to compare companies against each other directly** — Apple and JPMorgan file annual reports of very different lengths, which affects the underlying math in ways that make cross-company comparisons unreliable, even though within-company comparisons remain valid.
- **The system that separates quarterly earnings announcements from other company disclosures occasionally includes a few extra items it shouldn't**, for two of the five companies. This has been identified, measured, and is being tracked for correction — it affects roughly one additional filing per year for the companies involved. Because it's caught and disclosed here rather than silently producing a slightly-wrong trend line, users can weigh the affected sections of that chart accordingly.
- **The AI reading module in this system was not fine-tuned specifically on SEC filing language** — that refinement step is planned but requires a decision on the right training approach first, to avoid the tool essentially "grading its own homework."

## What This Demonstrates

Beyond the working tool itself, this project reflects a working methodology: every limitation listed above was found through direct verification (auditing actual output counts, testing specific edge cases, cross-checking against the raw source filings) rather than assumed to be fine. Where a fix was findable, it was made. Where a limitation is a genuine, inherent property of the method rather than a bug, it's disclosed rather than hidden or glossed over. That distinction — between "this is a known trade-off" and "this is a bug we haven't fixed yet" — is maintained throughout rather than blurred, which is what makes the tool's outputs trustworthy to build on.
