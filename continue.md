### Phase 2 status update (as of 2025-08-10)

- Personal Knowledge Graph
  - Compute clusters from existing embeddings with KMeans fallback: Complete
  - Create topics table with top keywords and representative conversations: Complete (centroid vectors are optional and currently omitted)
  - Text similarity search via wrapper: Complete

- Learning Gap Analysis
  - SQL views to surface confusion themes: Complete (view_confusion_signals, view_conversation_confusion)
  - Notebook cells to run and visualize gap analysis: Pending

- Code Pattern Library
  - code_snippets table creation and extraction from message content: Complete
  - Tagging via technical_terms heuristic persisted to tags field: Complete
  - Retrieval/query utilities and example notebook cells: Pending

- AI Assistant Optimization
  - Notebook comparisons (ChatGPT vs Claude) using sentiment, complexity, idea density: Pending
  - Routing guidelines artifact: Pending

- Timestamp handling improvements
  - Normalization in CSV: normalize_chats.py converts timestamps to ISO 8601 UTC (Z) with robust parsing and timezone safety
  - Database normalization: process_runner.py calls analyzer.normalize_db_timestamps() to ensure canonical UTC (Z) in DB after ingest

### What to do next (actionable checklist)

1) Run Phase 2 helpers on an existing DB
- Create/update gap-analysis SQL views
- Extract code snippets
- Build simple topics (KMeans fallback)

Commands:
- `python phase2_helpers.py --db chat_analysis.db --create-views`
- `python phase2_helpers.py --db chat_analysis.db --build-code-snippets`
- `python phase2_helpers.py --db chat_analysis.db --build-topics --n-clusters 12`

Tip: Run them separately for clearer diagnostics.

2) Expand the notebook to include Phase 2 sections
Add four new sections to `chat_analysis_notebook.ipynb`. Suggested cell outlines below (paste into new cells):

- Section: Phase 2 setup (utilities import)
  - `from phase2_helpers import create_gap_analysis_views, upsert_code_snippets, build_simple_topics, search_by_text`
  - `db_path = 'chat_analysis.db'`
  - `create_gap_analysis_views(db_path)`

- Section: Code Pattern Library (queries and examples)
  - `import sqlite3, pandas as pd, json`
  - `conn = sqlite3.connect(db_path)`
  - `df_snips = pd.read_sql_query("SELECT id, source_ai, conversation_id, snippet_type, language, substr(context_preview,1,200) AS preview, length(code) AS code_len, tags FROM code_snippets ORDER BY id DESC LIMIT 50", conn)`
  - Display top languages and snippet types (value_counts), parse tags: `df_snips.assign(tag_list=df_snips['tags'].apply(lambda t: json.loads(t) if t else []))`
  - Show a few representative snippets with preview

- Section: Topics overview
  - `df_topics = pd.read_sql_query("SELECT topic_id, top_keywords, representative_message_id, size FROM topics ORDER BY size DESC", conn)`
  - Join representative messages: `SELECT r.message_id, r.content FROM raw_conversations r WHERE r.message_id IN (...)`
  - Optional: bar chart of topic sizes

- Section: Learning Gap Analysis
  - `df_conf_sig = pd.read_sql_query("SELECT * FROM view_confusion_signals ORDER BY sentiment_score ASC LIMIT 200", conn)`
  - `df_conv_conf = pd.read_sql_query("SELECT * FROM view_conversation_confusion ORDER BY high_question_msgs DESC, avg_sentiment ASC LIMIT 100", conn)`
  - Visualize top conversations by high_question_msgs and low avg_sentiment; join titles from conversation_features for context

- Section: Text semantic search
  - `results = search_by_text(db_path, "vector database migration", limit=5)`
  - Pretty-print with conversation_id, role, source_ai, preview

3) Add AI assistant optimization analyses to the notebook
- Comparative charts and tables:
  - Sentiment by role and source_ai (pivot)
  - Complexity_score and idea_density distributions by source_ai
  - Code prevalence (has_code %) and question prevalence (has_questions %) by source_ai
- Add a concluding markdown cell with “routing guidance” based on observed metrics.

4) Update development_plan.md to reflect progress
Suggested edits for the Phase 2 block:
- Status: In Progress (2025-08-10)
- Personal Knowledge Graph: ✓ utilities/tables; Pending: visualization cells in notebook
- Learning Gap Analysis: ✓ SQL views; Pending: notebook cells for queries/visualizations
- Code Pattern Library: ✓ extraction & tags; Pending: retrieval utilities/notebook examples
- AI Assistant Optimization: Pending (comparative notebook cells + routing summary)
- Notes: Timestamp normalization improved in normalization script and enforced in DB post-ingest; notebook uses UTC-aware conversions

Acceptance criteria adjustments:
- Reproducible notebook cells that:
  - Re/create topics and views; upsert code_snippets from an existing DB
  - Query and visualize topics, code snippets, and gap analysis outputs
  - Run text semantic search and display top results with previews
  - Provide an “Assistant routing guidance” summary grounded in metrics

### Quick smoke run (no heavy models)
- If `chat_analysis.db` already exists, the Phase 2 helpers and SQL views do not require new heavy model downloads. KMeans requires scikit-learn; the helper prints a warning and skips if missing. If embeddings are absent, topics creation will skip with a clear message.

### Timestamp conversion notes and verification steps
- normalize_chats.py
  - ChatGPT: `datetime.fromtimestamp(..., tz=timezone.utc)` → ISO Z
  - Claude: `datetime.fromisoformat(... with 'Z'→'+00:00')`, ensure tz-aware, then ISO Z
- process_runner.py
  - Calls `analyzer.normalize_db_timestamps()` post-ingest to canonicalize UTC Z in DB even if upstream inputs vary
- Notebook
  - Uses `pd.to_datetime(..., utc=True, errors='coerce')` prior to time series; keep this pattern

Verification checklist:
- After running `process_runner.py` and before Phase 2 helpers:
  - `SELECT MIN(timestamp), MAX(timestamp) FROM raw_conversations;`
  - Ensure timestamps end with Z or are parsed as UTC in pandas

### Optional enhancements (nice-to-haves for Phase 2)
- topics: add a friendly `topic_label` derived from top keywords
- code_snippets: add `lines_of_code` and deduplicate identical snippets per message
- search_by_text: notebook helper to jump from message_id to full conversation transcript

### Summary
- Core Phase 2 helpers are implemented and ready to run.
- Next work: wire helpers into the notebook, execute them on your DB, and add comparisons for assistant optimization.
- Timestamp normalization is robust in both normalization and DB stages; maintain UTC Z end-to-end for consistent temporal analysis.