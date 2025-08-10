import sqlite3
import unittest

from chat_analysis_setup import ChatAnalyzer


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


class TestDBSetup(unittest.TestCase):
    def test_setup_database_in_memory(self):
        # Use in-memory DB and skip heavy model loading for a fast test
        analyzer = ChatAnalyzer(db_path=":memory:", load_models=False)

        conn = analyzer.conn
        self.assertTrue(table_exists(conn, 'raw_conversations'))
        self.assertTrue(table_exists(conn, 'message_features'))
        self.assertTrue(table_exists(conn, 'conversation_features'))
        self.assertTrue(table_exists(conn, 'embeddings'))


if __name__ == '__main__':
    unittest.main()
