import os
import json
import tempfile
import csv
import unittest
from typing import List

from normalize_chats import normalize_chatgpt_json, normalize_claude_json, merge_and_save_normalized_data


def _read_csv_headers(path: str) -> List[str]:
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        return next(reader)


class TestNormalizeChats(unittest.TestCase):
    def test_normalize_chatgpt_and_claude_and_merge(self):
        # Tiny, synthetic inputs
        chatgpt_payload = [
            {
                "id": "convo1",
                "title": "Test ChatGPT",
                "mapping": {
                    "node1": {
                        "message": {
                            "id": "m1",
                            "author": {"role": "user"},
                            "create_time": 1700000000,
                            "content": {"content_type": "text", "parts": ["hi there"]}
                        }
                    },
                    "node2": {
                        "message": {
                            "id": "m2",
                            "author": {"role": "assistant"},
                            "create_time": 1700000100,
                            "content": {"content_type": "text", "parts": ["hello back"]}
                        }
                    }
                }
            }
        ]

        claude_payload = [
            {
                "uuid": "c1",
                "name": "Test Claude",
                "chat_messages": [
                    {
                        "uuid": "x",
                        "sender": "assistant",
                        "created_at": "2023-11-14T12:00:00Z",
                        "content": [
                            {"type": "text", "text": "hello"}
                        ]
                    }
                ]
            }
        ]

        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as a, \
             tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as b:
            json.dump(chatgpt_payload, a); ap = a.name
            json.dump(claude_payload, b); bp = b.name

        with tempfile.TemporaryDirectory() as d:
            try:
                # Function-level checks
                chatgpt_rows = normalize_chatgpt_json(ap)
                claude_rows = normalize_claude_json(bp)

                self.assertEqual(len(chatgpt_rows), 2)
                self.assertEqual({r["Role"] for r in chatgpt_rows}, {"User", "Assistant"})
                self.assertEqual(len(claude_rows), 1)
                self.assertEqual(claude_rows[0]["Role"], "Assistant")

                # Merge and save
                out_csv = os.path.join(d, 'out.csv')
                merge_and_save_normalized_data(ap, bp, out_csv)
                self.assertTrue(os.path.exists(out_csv))

                headers = _read_csv_headers(out_csv)
                required = ["Source AI", "Conversation ID", "Conversation Title", "Message ID", "Timestamp", "Role", "Content", "Word Count"]
                for col in required:
                    self.assertIn(col, headers)
            finally:
                os.unlink(ap)
                os.unlink(bp)


if __name__ == '__main__':
    unittest.main()
