import tempfile
import unittest
from pathlib import Path

from agent.session_store import SessionStore


class TestSessionStore(unittest.TestCase):
    def test_save_and_load_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(str(db_path), max_sessions=10)

            session_id = "abc123"
            store.ensure_session(session_id, "llama3.2")
            store.save_message(session_id, "user", "hello", created_at="2024-01-01 00:00:00 UTC")
            store.save_tool_call(
                session_id=session_id,
                tool_name="response",
                args={"text": "hi"},
                result="hi",
                created_at="2024-01-01 00:00:01 UTC",
            )

            data = store.get_session(session_id)
            self.assertIsNotNone(data)
            self.assertEqual(data["session_id"], session_id)
            self.assertEqual(data["title"], "hello")
            self.assertEqual(len(data["messages"]), 1)
            self.assertEqual(data["messages"][0]["content"], "hello")
            self.assertEqual(len(data["tool_calls"]), 1)
            self.assertEqual(data["tool_calls"][0]["tool_name"], "response")

    def test_list_and_delete_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(str(db_path), max_sessions=10)

            store.ensure_session("s1", "model-a")
            store.ensure_session("s2", "model-b")

            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 2)

            store.set_title("s1", "Session One")
            sessions = store.list_sessions()
            titles = {s["session_id"]: s.get("title") for s in sessions}
            self.assertEqual(titles.get("s1"), "Session One")

            deleted = store.delete_session("s1")
            self.assertTrue(deleted)
            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)

    def test_prune_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "sessions.db"
            store = SessionStore(str(db_path), max_sessions=1)

            store.ensure_session("s1", "model-a")
            store.save_message("s1", "user", "hello", created_at="2024-01-01 00:00:00 UTC")
            store.ensure_session("s2", "model-b")
            store.save_message("s2", "user", "world", created_at="2024-01-01 00:00:01 UTC")

            sessions = store.list_sessions()
            self.assertEqual(len(sessions), 1)
