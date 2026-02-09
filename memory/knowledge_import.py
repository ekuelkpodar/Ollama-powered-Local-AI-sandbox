"""RAG pipeline — import documents into FAISS knowledge base."""

import hashlib
import json
import os

from memory.memory_manager import MemoryManager, AREA_KNOWLEDGE


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".html", ".json"}
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class KnowledgeImporter:
    """Scans directories for documents, chunks them, and stores in FAISS."""

    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self._state_file = os.path.join(
            memory_manager.config.data_dir, "knowledge_import.json"
        )
        self._state = self._load_state()

    def _load_state(self) -> dict:
        """Load import state (file hashes) from disk."""
        if os.path.exists(self._state_file):
            with open(self._state_file, "r") as f:
                return json.load(f)
        return {}

    def _save_state(self):
        """Persist import state to disk."""
        os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
        with open(self._state_file, "w") as f:
            json.dump(self._state, f, indent=2)

    async def import_directory(
        self, directory: str | None = None, subdir: str = "default"
    ) -> dict:
        """
        Import all supported files from a directory.
        Returns stats: {"imported": N, "skipped": N, "errors": [...]}
        """
        if directory is None:
            directory = self.memory_manager.config.knowledge_dir
        os.makedirs(directory, exist_ok=True)

        stats = {"imported": 0, "skipped": 0, "errors": []}

        for root, _, files in os.walk(directory):
            for filename in sorted(files):
                ext = os.path.splitext(filename)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    continue

                filepath = os.path.join(root, filename)
                file_hash = self._file_md5(filepath)

                if self._state.get(filepath) == file_hash:
                    stats["skipped"] += 1
                    continue

                try:
                    text = self._parse_file(filepath, ext)
                    chunks = self._chunk_text(text)

                    for i, chunk in enumerate(chunks):
                        await self.memory_manager.save(
                            content=chunk,
                            area=AREA_KNOWLEDGE,
                            metadata={
                                "source": filepath,
                                "chunk_index": i,
                                "filename": filename,
                            },
                            subdir=subdir,
                        )

                    self._state[filepath] = file_hash
                    stats["imported"] += 1
                except Exception as e:
                    stats["errors"].append(f"{filename}: {e}")

        self._save_state()
        return stats

    def _parse_file(self, filepath: str, ext: str) -> str:
        """Extract text content from a file based on its extension."""
        if ext in (".txt", ".md"):
            with open(filepath, "r", errors="replace") as f:
                return f.read()

        if ext == ".pdf":
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(filepath)
                pages = [page.extract_text() or "" for page in reader.pages]
                return "\n\n".join(pages)
            except ImportError:
                with open(filepath, "r", errors="replace") as f:
                    return f.read()

        if ext == ".html":
            try:
                from bs4 import BeautifulSoup
                with open(filepath, "r", errors="replace") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                return soup.get_text(separator="\n")
            except ImportError:
                with open(filepath, "r", errors="replace") as f:
                    return f.read()

        if ext == ".csv":
            with open(filepath, "r", errors="replace") as f:
                return f.read()

        if ext == ".json":
            with open(filepath, "r") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)

        with open(filepath, "r", errors="replace") as f:
            return f.read()

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        step = CHUNK_SIZE - CHUNK_OVERLAP
        for i in range(0, len(text), step):
            chunk = text[i:i + CHUNK_SIZE].strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text.strip()] if text.strip() else []

    def _file_md5(self, filepath: str) -> str:
        """Compute MD5 hash of a file."""
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()
