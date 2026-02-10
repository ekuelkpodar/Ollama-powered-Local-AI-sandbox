import tempfile
import unittest
from pathlib import Path

from agent.output_parser import OutputParser


class TestOutputParser(unittest.TestCase):
    def _parser(self, log_dir: Path) -> OutputParser:
        return OutputParser(str(log_dir))

    def test_extract_from_code_fence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = self._parser(Path(tmpdir))
            raw = "Here you go:\n```json\n{\"tool_name\":\"response\",\"tool_args\":{\"text\":\"hi\"}}\n```"
            tool_call = parser.extract_tool_call(
                raw_text=raw,
                registered_tools=["response"],
                tool_schemas={"response": {"required_args": ["text"], "arg_schema": {"text": str}}},
            )
            self.assertIsNotNone(tool_call)
            self.assertEqual(tool_call.name, "response")
            self.assertEqual(tool_call.args.get("text"), "hi")

    def test_extract_first_valid_tool(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = self._parser(Path(tmpdir))
            raw = (
                "```json\n{'tool_name':'unknown','tool_args':{}}\n```\n"
                "```json\n{\"tool_name\":\"response\",\"tool_args\":{\"text\":\"ok\"}}\n```"
            )
            tool_call = parser.extract_tool_call(
                raw_text=raw,
                registered_tools=["response"],
                tool_schemas={"response": {"required_args": ["text"], "arg_schema": {"text": str}}},
            )
            self.assertIsNotNone(tool_call)
            self.assertEqual(tool_call.args.get("text"), "ok")

    def test_repair_single_quotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = self._parser(Path(tmpdir))
            raw = "{'tool_name':'response','tool_args':{'text':'hello'}}"
            tool_call = parser.extract_tool_call(
                raw_text=raw,
                registered_tools=["response"],
                tool_schemas={"response": {"required_args": ["text"], "arg_schema": {"text": str}}},
            )
            self.assertIsNotNone(tool_call)
            self.assertEqual(tool_call.args.get("text"), "hello")

    def test_rejects_missing_required_args(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = self._parser(Path(tmpdir))
            raw = "{\"tool_name\":\"response\",\"tool_args\":{}}"
            tool_call = parser.extract_tool_call(
                raw_text=raw,
                registered_tools=["response"],
                tool_schemas={"response": {"required_args": ["text"], "arg_schema": {"text": str}}},
            )
            self.assertIsNone(tool_call)

    def test_extract_multiple_tools_from_array(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = self._parser(Path(tmpdir))
            raw = (
                "```json\n["
                "{\"tool_name\":\"memory\",\"tool_args\":{\"action\":\"search\",\"text\":\"hi\"}},"
                "{\"tool_name\":\"response\",\"tool_args\":{\"text\":\"done\"}}"
                "]\n```"
            )
            tool_calls = parser.extract_tool_calls(
                raw_text=raw,
                registered_tools=["memory", "response"],
                tool_schemas={
                    "memory": {"required_args": [], "arg_schema": {"action": str, "text": str}},
                    "response": {"required_args": ["text"], "arg_schema": {"text": str}},
                },
            )
            self.assertEqual(len(tool_calls), 2)
            self.assertEqual(tool_calls[0].name, "memory")
            self.assertEqual(tool_calls[1].name, "response")

    def test_logs_parse_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir)
            parser = self._parser(log_dir)
            raw = "no tools here"
            tool_call = parser.extract_tool_call(
                raw_text=raw,
                registered_tools=["response"],
                tool_schemas={"response": {"required_args": ["text"], "arg_schema": {"text": str}}},
            )
            self.assertIsNone(tool_call)
            log_path = log_dir / "output_parser.log"
            self.assertTrue(log_path.exists())
            content = log_path.read_text(encoding="utf-8")
            self.assertIn("RAW_OUTPUT", content)
            self.assertIn(raw, content)
