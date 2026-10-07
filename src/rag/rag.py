"""
rag.py — Course-level RAG (Retrieval-Augmented Generation) for lecture context

Uses ChromaDB for vector storage. Embeddings: EmbeddingGemma 2 (src/slides/eg2.py: multilingual,
8K context, task prefixes "Document" / "SearchQuery") when torch + sentence-transformers are
installed, otherwise ChromaDB's bundled ONNX all-MiniLM-L6-v2 (English only, no torch needed).
Each collection records its embedder in the metadata: vectors of different models never mix
(tools/rag_migrate_eg2.py re-embeds an old MiniLM database).
Allows querying previous lectures to provide context for notes generation.
"""

from pathlib import Path

from rich.console import Console

console = Console()

# Header of each retrieved passage in the prompt. Unknown sources: "[Lecture <date>, <source>]"
SOURCE_LABELS = {
    "transcript": "[Lecture {date}, transcript]",
    "pdf": "[Lecture {date}, our notes: {doc}]",
    "handout": "[Official course material (handout): {doc}]",
    "third-party": ("[Notes written by ANOTHER STUDENT: {doc} — not authoritative, may contain "
                    "errors; the lecture transcript and the slides take precedence]"),
}


class CourseRAG:
    """
    RAG system for indexing and querying lecture transcripts by course.
    Uses ChromaDB for persistent vector storage and sentence-transformers for embeddings.
    """

    def __init__(self, db_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for RAG. Install it with: pip install chromadb"
            )

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        from src.slides import eg2
        if eg2.available():
            self.embedder = "embeddinggemma-2"
            self._embed_docs = lambda texts: eg2.encode_texts(texts, prompt_name="Document").tolist()
            self._embed_query = lambda texts: eg2.encode_texts(texts, prompt_name="SearchQuery").tolist()
        else:
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            ef = DefaultEmbeddingFunction()          # all-MiniLM-L6-v2 via onnxruntime, CPU
            self.embedder = "all-MiniLM-L6-v2"
            self._embed_docs = self._embed_query = lambda texts: [list(map(float, v)) for v in ef(texts)]

    def _get_collection(self, course_name: str):
        """Get or create a ChromaDB collection for a course."""
        safe_name = course_name.lower().replace(" ", "_").replace("-", "_")
        # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        safe_name = safe_name[:63] if len(safe_name) > 63 else safe_name
        if len(safe_name) < 3:
            safe_name = safe_name + "_course"
        col = self.client.get_or_create_collection(name=safe_name, metadata={"embedder": self.embedder})
        have = (col.metadata or {}).get("embedder", "all-MiniLM-L6-v2")
        if have != self.embedder:
            raise RuntimeError(f"RAG: collection '{safe_name}' in {self.db_path} was built with {have}, "
                               f"this run embeds with {self.embedder} (install torch + sentence-transformers, "
                               f"or migrate with tools/rag_migrate_eg2.py)")
        return col

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        for i in range(0, len(words), step):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        return chunks

    def add_lecture(
        self,
        transcript: str,
        course_name: str,
        lecture_date: str,
        source: str = "transcript",
        doc: str | None = None,
    ) -> int:
        """
        Index a lecture transcript into the RAG database.
        Returns the number of chunks added.
        doc: a document key (e.g. the PDF file name) for material that is not one-per-date:
        ids come from it instead of the date, and its previous chunks are replaced.
        """
        collection = self._get_collection(course_name)
        chunks = self._chunk_text(transcript)
        if doc:
            collection.delete(where={"doc": doc})

        if not chunks:
            console.print("[dim]RAG: no chunks to index[/dim]")
            return 0

        # Generate embeddings
        embeddings = self._embed_docs(chunks)

        # Create unique IDs based on course, date, and chunk index
        key = f"doc:{doc}" if doc else lecture_date
        ids = [
            f"{course_name}_{key}_{source}_{i}"
            for i in range(len(chunks))
        ]

        # Store metadata
        metadatas = [
            {
                "course": course_name,
                "lecture_date": lecture_date,
                "source": source,
                "chunk_index": i,
                **({"doc": doc} if doc else {}),
            }
            for i in range(len(chunks))
        ]

        # Upsert to handle re-indexing
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        console.print(
            f"[green]✓ RAG: indexed {len(chunks)} chunks[/green] "
            f"(course={course_name}, date={lecture_date}, source={source})"
        )
        return len(chunks)

    def query_context(
        self,
        query_text: str | list[str],
        course_name: str,
        n_results: int = 5,
        exclude_date: str | None = None,
        exclude_doc: str | None = None,
    ) -> str:
        """
        Query the RAG database for relevant context from previous lectures.
        query_text may be a list (e.g. samples from the start, middle and end of the lecture):
        results are merged by distance, one entry per chunk. exclude_date leaves out the
        lecture being generated (already indexed when notes are regenerated); exclude_doc does it
        by document, so other lectures of the same day and other people's notes of this very
        lecture stay in.
        Returns a formatted string of relevant passages with their lecture date.
        """
        collection = self._get_collection(course_name)

        if collection.count() == 0:
            return ""

        queries = [query_text] if isinstance(query_text, str) else [q for q in query_text if q.strip()]
        if not queries:
            return ""
        kwargs = ({"where": {"doc": {"$ne": exclude_doc}}} if exclude_doc else
                  {"where": {"lecture_date": {"$ne": exclude_date}}} if exclude_date else {})
        results = collection.query(
            query_embeddings=self._embed_query(queries),
            n_results=min(n_results, collection.count()),
            **kwargs,
        )

        best: dict[str, tuple[float, str, dict]] = {}
        for ids, docs, metas, dists in zip(results["ids"], results["documents"], results["metadatas"], results["distances"]):
            for i, doc, meta, dist in zip(ids, docs, metas, dists):
                if i not in best or dist < best[i][0]:
                    best[i] = (dist, doc, meta)
        if not best:
            return ""
        top = sorted(best.values(), key=lambda x: x[0])[:n_results]
        # in ordine di lezione e di posizione, così il contesto si legge come materiale del corso
        top.sort(key=lambda x: (x[2].get("lecture_date", ""), x[2].get("chunk_index", 0)))

        passages = []
        for _, doc, metadata in top:
            date = metadata.get("lecture_date", "unknown")
            source = metadata.get("source", "transcript")
            passages.append(f"{SOURCE_LABELS.get(source, '[Lecture {date}, {source}]').format(date=date, doc=metadata.get('doc', ''))}\n{doc}")

        context = "\n\n---\n\n".join(passages)
        console.print(f"[dim]RAG: retrieved {len(passages)} passages from previous lectures[/dim]")
        return context

    def is_indexed(self, course_name: str, lecture_date: str, source: str = "transcript",
                   doc: str | None = None) -> bool:
        """True if at least one chunk of that lecture (of that document, with doc) is in the index."""
        try:
            where = ({"doc": doc} if doc else
                     {"$and": [{"lecture_date": lecture_date}, {"source": source}]})
            got = self._get_collection(course_name).get(where=where, limit=1)
            return bool(got["ids"])
        except Exception:
            return False

    def add_from_pdf(self, pdf_path: Path, course_name: str, source: str = "pdf") -> int:
        """
        Extract text from a PDF and add it to the RAG database.
        source: "pdf" (our own notes), "handout" (official material), "third-party" (other students'
        notes): it decides how the passages are labelled in the prompt. The file name is the
        document key, so re-indexing a file replaces it and two PDFs never overwrite each other.
        Returns the number of chunks added.
        """
        import re
        import pymupdf
        try:
            with pymupdf.open(pdf_path) as pdf:
                text_parts = [t for page in pdf if (t := page.get_text().strip())]
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to read PDF {pdf_path}: {e}[/yellow]")
            return 0

        if not text_parts:
            console.print(f"[yellow]⚠ RAG: no text in {pdf_path.name} (scanned? OCR is not supported)[/yellow]")
            return 0

        # date from the file name (DD-MM-YYYY… or YYYY-MM-DD…), as ISO like the transcripts
        lecture_date = ""
        if m := re.match(r"(\d{2})-(\d{2})-(\d{4})", pdf_path.name):
            lecture_date = f"{m[3]}-{m[2]}-{m[1]}"
        elif m := re.match(r"\d{4}-\d{2}-\d{2}", pdf_path.name):
            lecture_date = m[0]

        return self.add_lecture(
            transcript="\n\n".join(text_parts),
            course_name=course_name,
            lecture_date=lecture_date,
            source=source,
            doc=pdf_path.stem,
        )

    def course_exists(self, course_name: str) -> bool:
        """Check if any documents exist for this course."""
        try:
            collection = self._get_collection(course_name)
            return collection.count() > 0
        except Exception:
            return False
