import os
import sqlite3
import unittest
from typing import Tuple

from phase2_helpers import upsert_code_snippets, create_gap_analysis_views


def _make_minimal_db() -> Tuple[str, sqlite3.Connection]:
    # Create a temporary file-backed DB to persist across connections
    db_path = os.path.abspath('test_phase2.db')
    try:
        os.remove(db_path)
    except FileNotFoundError:
        pass
    conn = sqlite3.connect(db_path)

    # Minimal schema required by helpers
    conn.execute(
        """
        CREATE TABLE raw_conversations (
            message_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            source_ai TEXT,
            timestamp TEXT,
            role TEXT,
            content TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE message_features (
            message_id TEXT PRIMARY KEY,
            has_code INTEGER,
            sentiment_score REAL,
            question_count INTEGER,
            content_type TEXT,
            clean_content TEXT,
            technical_terms TEXT
        )
        """
    )

    # Insert one code-heavy message
    content = (
        "Here is some code example.\n"
        "```python\nprint('hello')\n```\n"
        "And some `inline()` call."
    )
    conn.execute(
        "INSERT INTO raw_conversations (message_id, conversation_id, source_ai, timestamp, role, content) VALUES (?, ?, ?, datetime('now'), ?, ?)",
        ("m1", "c1", "ChatGPT", "user", content)
    )
    conn.execute(
        "INSERT INTO message_features (message_id, has_code, sentiment_score, question_count, content_type, clean_content, technical_terms) VALUES (?, 1, 0.1, 2, 'debug', ?, ?)",
        ("m1", content, "[\"python\", \"example\"]")
    )

    # Add two more messages for the same conversation to satisfy HAVING msg_count >= 3
    conn.execute(
        "INSERT INTO raw_conversations (message_id, conversation_id, source_ai, timestamp, role, content) VALUES ('m2', 'c1', 'ChatGPT', datetime('now'), 'assistant', 'Reply text')"
    )
    conn.execute(
        "INSERT INTO message_features (message_id, has_code, sentiment_score, question_count, content_type, clean_content, technical_terms) VALUES ('m2', 0, 0.8, 0, 'explanation', 'Reply text', '[]')"
    )
    conn.execute(
        "INSERT INTO raw_conversations (message_id, conversation_id, source_ai, timestamp, role, content) VALUES ('m3', 'c1', 'ChatGPT', datetime('now'), 'user', 'Another question?')"
    )
    conn.execute(
        "INSERT INTO message_features (message_id, has_code, sentiment_score, question_count, content_type, clean_content, technical_terms) VALUES ('m3', 0, 0.2, 2, 'question', 'Another question?', '[]')"
    )

    conn.commit()
    return db_path, conn


class TestPhase2Helpers(unittest.TestCase):
    def test_code_snippets_and_views(self):
        db_path, conn = _make_minimal_db()
        try:
            # Run extraction
            inserted = upsert_code_snippets(db_path)
            self.assertGreaterEqual(inserted, 2)  # one block + one inline

            # Ensure table exists and has rows
            cur = conn.execute("SELECT COUNT(*) FROM code_snippets")
            count = cur.fetchone()[0]
            self.assertGreaterEqual(count, 2)

            # Create views
            create_gap_analysis_views(db_path)
            # Check views exist by querying
            cur = conn.execute("SELECT COUNT(*) FROM view_confusion_signals")
            self.assertGreaterEqual(cur.fetchone()[0], 1)

            cur = conn.execute("SELECT * FROM view_conversation_confusion WHERE conversation_id = 'c1'")
            rows = cur.fetchall()
            self.assertTrue(len(rows) >= 1)
        finally:
            conn.close()
            try:
                os.remove(db_path)
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    unittest.main()
