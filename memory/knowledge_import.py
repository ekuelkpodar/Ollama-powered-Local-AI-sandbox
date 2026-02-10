"""RAG pipeline — import documents into FAISS knowledge base."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone

from memory.memory_manager import MemoryManager, AREA_KNOWLEDGE


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".html", ".json"}


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
        self,
        directory: str | None = None,
        namespace: str | None = None,
        subdir: str = "default",
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
                    base_meta = self._extract_metadata(text, filepath, ext)

                    for i, chunk in enumerate(chunks):
                        await self.memory_manager.save(
                            content=chunk,
                            area=AREA_KNOWLEDGE,
                            metadata={
                                **base_meta,
                                "chunk_index": i,
                                "chunk_count": len(chunks),
                                "chunk_size": self._chunk_size(),
                                "chunk_overlap": self._chunk_overlap(),
                                "chunk_strategy": self._chunk_strategy(),
                                "ingested_at": datetime.now(timezone.utc).isoformat(),
                            },
                            namespace=namespace,
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
        strategy = self._chunk_strategy()
        if strategy == "sentence":
            chunks = self._chunk_by_sentence(text)
        elif strategy == "semantic":
            chunks = self._chunk_by_semantic(text)
        else:
            chunks = self._chunk_by_fixed(text)
        return self._apply_overlap(chunks)

    def _chunk_by_fixed(self, text: str) -> list[str]:
        chunks: list[str] = []
        size = self._chunk_size()
        overlap = self._chunk_overlap()
        step = max(size - overlap, 1)
        for i in range(0, len(text), step):
            chunk = text[i:i + size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text.strip()] if text.strip() else []

    def _chunk_by_sentence(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        size = self._chunk_size()
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if not sentence:
                continue
            sentence_len = len(sentence)
            if current and current_len + sentence_len > size:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = sentence_len
            else:
                current.append(sentence)
                current_len += sentence_len
        if current:
            chunks.append(" ".join(current).strip())
        return chunks if chunks else [text.strip()] if text.strip() else []

    def _chunk_by_semantic(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [text.strip()] if text.strip() else []
        embeddings = self.memory_manager.embedding_engine.embed(sentences)
        size = self._chunk_size()
        threshold = self._semantic_threshold()
        chunks: list[str] = []
        current: list[str] = [sentences[0]]
        current_len = len(sentences[0])
        for idx in range(1, len(sentences)):
            sentence = sentences[idx]
            similarity = float(embeddings[idx - 1].dot(embeddings[idx]))
            if current_len + len(sentence) > size or similarity < threshold:
                chunks.append(" ".join(current).strip())
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += len(sentence)
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        overlap = self._chunk_overlap()
        if overlap <= 0 or len(chunks) <= 1:
            return chunks
        overlapped = [chunks[0]]
        for idx in range(1, len(chunks)):
            prev = overlapped[-1]
            prefix = prev[-overlap:] if len(prev) > overlap else prev
            combined = (prefix + " " + chunks[idx]).strip()
            overlapped.append(combined)
        return overlapped

    def _split_sentences(self, text: str) -> list[str]:
        raw = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _chunk_strategy(self) -> str:
        return self.memory_manager.config.rag.chunk_strategy

    def _chunk_size(self) -> int:
        return self.memory_manager.config.rag.chunk_size

    def _chunk_overlap(self) -> int:
        return self.memory_manager.config.rag.chunk_overlap

    def _semantic_threshold(self) -> float:
        return self.memory_manager.config.rag.semantic_threshold

    def _extract_metadata(self, text: str, filepath: str, ext: str) -> dict:
        """Extract metadata from document content."""
        filename = os.path.basename(filepath)
        html_text = None
        if ext == ".html":
            try:
                with open(filepath, "r", errors="replace") as f:
                    html_text = f.read()
            except Exception:
                html_text = None

        title = self._extract_title(text, filename, ext, html_text=html_text)
        date = self._extract_date(text, filename)
        headers = self._extract_section_headers(text, ext, html_text=html_text)
        return {
            "source": filepath,
            "filename": filename,
            "title": title,
            "date": date,
            "section_headers": headers,
        }

    def _extract_title(self, text: str, filename: str, ext: str, html_text: str | None = None) -> str:
        if ext == ".md":
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    return line.lstrip("#").strip()
        if ext == ".html":
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_text or text, "html.parser")
                if soup.title and soup.title.text:
                    return soup.title.text.strip()
                h1 = soup.find("h1")
                if h1 and h1.text:
                    return h1.text.strip()
            except Exception:
                pass
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:200]
        return os.path.splitext(filename)[0]

    def _extract_date(self, text: str, filename: str) -> str | None:
        patterns = [
            r"\b(20\d{2}-\d{2}-\d{2})\b",
            r"\b(20\d{2}/\d{2}/\d{2})\b",
            r"(?i)date:\s*(20\d{2}-\d{2}-\d{2})",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", filename)
        if match:
            return match.group(1)
        return None

    def _extract_section_headers(self, text: str, ext: str, html_text: str | None = None) -> list[str]:
        headers: list[str] = []
        if ext == ".md":
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("#"):
                    header = line.lstrip("#").strip()
                    if header:
                        headers.append(header)
        elif ext == ".html":
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html_text or text, "html.parser")
                for level in ("h1", "h2", "h3"):
                    for tag in soup.find_all(level):
                        if tag.text:
                            headers.append(tag.text.strip())
            except Exception:
                pass
        return headers[:20]

    def _file_md5(self, filepath: str) -> str:
        """Compute MD5 hash of a file."""
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()
