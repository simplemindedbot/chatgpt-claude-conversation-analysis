"""
Phase 2 helpers — Analysis Enablement utilities

Implements lightweight, dependency-friendly utilities to:
- Build a code_snippets table from raw_conversations/message_features.
- Optionally compute simple topic clusters from existing embeddings and persist a topics table.
- Create helper SQL views for learning gap analysis.
- Provide a simple search-by-text helper delegating to SemanticSearchEngine.

Follows the repo guidelines:
- Gracefully degrade if optional deps (UMAP/HDBSCAN/BERTopic/sqlite-vec) are unavailable.
- Keep unit-testability: pure-ish functions that operate on SQLite DB paths and avoid heavy model downloads.
"""
from __future__ import annotations

import re
import json
import sqlite3
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional ML imports kept lightweight
try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover - optional at runtime
    KMeans = None  # type: ignore
    TfidfVectorizer = None  # type: ignore


CODE_BLOCK_RE = re.compile(r"```(\w+)?\n(.*?)\n```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]+)`")


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    # Ensures text is returned as str
    conn.text_factory = str
    return conn


def ensure_code_snippets_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS code_snippets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            conversation_id TEXT,
            source_ai TEXT,
            snippet_type TEXT,   -- 'block' or 'inline'
            language TEXT,       -- heuristic from ```lang```
            code TEXT,
            context_preview TEXT,
            sentiment_score REAL,
            tags TEXT            -- JSON array of tags/tech terms
        )
        """
    )
    conn.commit()


def upsert_code_snippets(db_path: str, limit: Optional[int] = None) -> int:
    """
    Extract code from messages with has_code flag and store into code_snippets table.
    Returns number of inserted rows.
    """
    conn = _connect(db_path)
    ensure_code_snippets_table(conn)

    query = (
        """
        SELECT r.message_id, r.conversation_id, r.source_ai, r.content,
               COALESCE(f.sentiment_score, 0.0) AS sentiment_score,
               f.technical_terms
        FROM raw_conversations r
        JOIN message_features f ON r.message_id = f.message_id
        WHERE f.has_code = 1 AND r.content IS NOT NULL AND length(r.content) > 0
        ORDER BY r.timestamp ASC
        """
    )
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return 0

    inserts = []
    for _, row in df.iterrows():
        content: str = row["content"] or ""
        context_preview = content[:240]
        tags: List[str] = []
        try:
            if row["technical_terms"]:
                terms = json.loads(row["technical_terms"]) if isinstance(row["technical_terms"], str) else row["technical_terms"]
                if isinstance(terms, list):
                    tags = terms
        except Exception:
            pass

        # Block code
        for m in CODE_BLOCK_RE.finditer(content):
            lang = (m.group(1) or "unknown").lower().strip()
            code = m.group(2).strip()
            if code:
                inserts.append(
                    (
                        row["message_id"], row["conversation_id"], row["source_ai"],
                        "block", lang, code, context_preview, float(row["sentiment_score"]), json.dumps(tags)
                    )
                )
        # Inline code
        for m in INLINE_CODE_RE.finditer(content):
            code = m.group(1).strip()
            if code:
                inserts.append(
                    (
                        row["message_id"], row["conversation_id"], row["source_ai"],
                        "inline", "unknown", code, context_preview, float(row["sentiment_score"]), json.dumps(tags)
                    )
                )

        if limit and len(inserts) >= limit:
            break

    if not inserts:
        return 0

    conn.executemany(
        """
        INSERT INTO code_snippets (
            message_id, conversation_id, source_ai, snippet_type, language, code, context_preview, sentiment_score, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        inserts,
    )
    conn.commit()
    return len(inserts)


def ensure_topics_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS topics (
            topic_id INTEGER,
            top_keywords TEXT,           -- comma-separated keywords
            representative_message_id TEXT,
            size INTEGER,
            PRIMARY KEY(topic_id)
        )
        """
    )
    conn.commit()


def _load_embeddings_with_text(conn: sqlite3.Connection) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_sql_query(
        """
        SELECT e.message_id, e.embedding_vector, COALESCE(f.clean_content, r.content) AS text
        FROM embeddings e
        JOIN raw_conversations r ON e.message_id = r.message_id
        LEFT JOIN message_features f ON e.message_id = f.message_id
        WHERE e.embedding_vector IS NOT NULL AND (text IS NOT NULL AND length(text) > 0)
        """,
        conn,
    )
    if df.empty:
        return df, np.zeros((0,))
    embs = [np.frombuffer(b, dtype=np.float32) for b in df["embedding_vector"]]
    X = np.vstack(embs) if embs else np.zeros((0,))
    return df, X


def build_simple_topics(db_path: str, n_clusters: int = 12, max_docs: int = 5000) -> Optional[int]:
    """
    Build a lightweight topics table using KMeans over existing embeddings.
    Falls back gracefully if scikit-learn isn't available or no embeddings are present.
    Returns number of topics inserted, or None if skipped.
    """
    if KMeans is None:
        print("⚠️ scikit-learn not available; skipping topics table creation")
        return None

    conn = _connect(db_path)
    ensure_topics_table(conn)

    df, X = _load_embeddings_with_text(conn)
    if isinstance(X, np.ndarray) and X.size == 0:
        print("⚠️ No embeddings/text available; skipping topics")
        return None

    # Limit docs for performance
    if len(df) > max_docs:
        df = df.sample(n=max_docs, random_state=42).reset_index(drop=True)
        X = X[df.index]

    # Fit KMeans
    k = min(n_clusters, max(2, len(df) // 50))  # heuristic: at least 2, ~50 docs per cluster
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    df["topic_id"] = labels

    # Compute keywords by TF-IDF per cluster if available
    keywords_by_topic = {}
    if TfidfVectorizer is not None:
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
        tfidf = vectorizer.fit_transform(df["text"].fillna("").tolist())
        terms = np.array(vectorizer.get_feature_names_out())
        for t in sorted(df["topic_id"].unique()):
            idx = np.where(df["topic_id"] == t)[0]
            if idx.size == 0:
                keywords_by_topic[t] = []
                continue
            centroid_scores = np.asarray(tfidf[idx].mean(axis=0)).ravel()
            top_idx = centroid_scores.argsort()[-8:][::-1]
            keywords_by_topic[t] = terms[top_idx].tolist()
    else:
        for t in sorted(df["topic_id"].unique()):
            keywords_by_topic[t] = []

    # Representative message per topic (closest to cluster center in embedding space)
    reps = {}
    centers = km.cluster_centers_
    for t in range(k):
        idx = np.where(df["topic_id"] == t)[0]
        if idx.size == 0:
            continue
        Xc = X[idx]
        dists = np.linalg.norm(Xc - centers[t], axis=1)
        best = idx[int(dists.argmin())]
        reps[t] = df.loc[best, "message_id"]

    # Clear and insert topics
    conn.execute("DELETE FROM topics")
    rows = []
    for t in range(k):
        size = int((df["topic_id"] == t).sum())
        rows.append((int(t), ", ".join(keywords_by_topic.get(t, [])), reps.get(t, None), size))
    conn.executemany(
        "INSERT INTO topics (topic_id, top_keywords, representative_message_id, size) VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def create_gap_analysis_views(db_path: str) -> None:
    """Create helper SQL views for learning gap analysis (idempotent)."""
    conn = _connect(db_path)
    # View: messages likely indicating confusion
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS view_confusion_signals AS
        SELECT r.message_id, r.conversation_id, r.source_ai, r.content, f.sentiment_score, f.question_count,
               f.content_type, COALESCE(f.has_code, 0) AS has_code
        FROM raw_conversations r
        JOIN message_features f ON r.message_id = f.message_id
        WHERE f.question_count > 1 AND f.sentiment_score < 0.5
        """
    )
    # View: conversation-level confusion density
    conn.execute(
        """
        CREATE VIEW IF NOT EXISTS view_conversation_confusion AS
        SELECT r.conversation_id,
               AVG(f.sentiment_score) AS avg_sentiment,
               AVG(f.question_count) AS avg_questions,
               SUM(CASE WHEN f.question_count > 1 THEN 1 ELSE 0 END) AS high_question_msgs,
               COUNT(*) AS msg_count
        FROM raw_conversations r
        JOIN message_features f ON r.message_id = f.message_id
        GROUP BY r.conversation_id
        HAVING msg_count >= 3
        """
    )
    conn.commit()


# Thin wrapper around existing semantic search tool for convenience

def search_by_text(db_path: str, query: str, limit: int = 10) -> List[dict]:
    from semantic_search_tool import SemanticSearchEngine  # Local import to avoid heavy load during tests
    engine = SemanticSearchEngine(db_path=db_path)
    return engine.semantic_search_vec(query, limit=limit)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 2 utilities: topics, code snippets, and gap-analysis views")
    parser.add_argument("--db", dest="db", default="chat_analysis.db", help="Path to SQLite DB")
    parser.add_argument("--build-topics", action="store_true", help="Create/update topics table using KMeans")
    parser.add_argument("--n-clusters", type=int, default=12, help="Number of KMeans clusters (hint; adjusted heuristically)")
    parser.add_argument("--build-code-snippets", action="store_true", help="Extract and store code snippets into code_snippets table")
    parser.add_argument("--limit", type=int, default=0, help="Max snippets to insert (0 = no limit)")
    parser.add_argument("--create-views", action="store_true", help="Create helper SQL views for gap analysis")
    parser.add_argument("--search", type=str, default=None, help="Run a sample text search and print top results")

    args = parser.parse_args()
    db_path = args.db

    if args.create_views:
        create_gap_analysis_views(db_path)
        print("✅ Created/updated gap-analysis SQL views")

    if args.build_code_snippets:
        count = upsert_code_snippets(db_path, limit=args.limit or None)
        print(f"✅ Inserted {count} code snippets into code_snippets table")

    if args.build_topics:
        n = build_simple_topics(db_path, n_clusters=args.n_clusters)
        if n is None:
            print("⚠️ Skipped building topics (missing deps or data)")
        else:
            print(f"✅ Built topics table with {n} topics")

    if args.search:
        results = search_by_text(db_path, args.search, limit=5)
        print(f"🔎 Top results for '{args.search}': {len(results)}")
        for i, r in enumerate(results, 1):
            content = r.get("content", "")
            preview = (content[:140] + "...") if len(content) > 140 else content
            print(f"{i}. {r.get('source_ai')} | {r.get('role')} | conv {r.get('conversation_id')} | score ~ {1 - r.get('distance', 0):.3f}")
            print(f"   {preview}")
