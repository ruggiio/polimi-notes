"""
rag.py — Course-level RAG (Retrieval-Augmented Generation) for lecture context

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Allows querying previous lectures to provide context for notes generation.
"""

from pathlib import Path

from rich.console import Console

console = Console()


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
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG. Install it with: pip install sentence-transformers"
            )

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_collection(self, course_name: str):
        """Get or create a ChromaDB collection for a course."""
        safe_name = course_name.lower().replace(" ", "_").replace("-", "_")
        # ChromaDB collection names must be 3-63 chars, alphanumeric + underscores
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        safe_name = safe_name[:63] if len(safe_name) > 63 else safe_name
        if len(safe_name) < 3:
            safe_name = safe_name + "_course"
        return self.client.get_or_create_collection(name=safe_name)

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
    ) -> int:
        """
        Index a lecture transcript into the RAG database.
        Returns the number of chunks added.
        """
        collection = self._get_collection(course_name)
        chunks = self._chunk_text(transcript)

        if not chunks:
            console.print("[dim]RAG: no chunks to index[/dim]")
            return 0

        # Generate embeddings
        embeddings = self.model.encode(chunks).tolist()

        # Create unique IDs based on course, date, and chunk index
        ids = [
            f"{course_name}_{lecture_date}_{source}_{i}"
            for i in range(len(chunks))
        ]

        # Store metadata
        metadatas = [
            {
                "course": course_name,
                "lecture_date": lecture_date,
                "source": source,
                "chunk_index": i,
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
        query_text: str,
        course_name: str,
        n_results: int = 5,
    ) -> str:
        """
        Query the RAG database for relevant context from previous lectures.
        Returns a formatted string of relevant passages with their lecture date.
        """
        collection = self._get_collection(course_name)

        if collection.count() == 0:
            return ""

        query_embedding = self.model.encode([query_text]).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return ""

        passages = []
        for doc, metadata in zip(results["documents"][0], results["metadatas"][0]):
            date = metadata.get("lecture_date", "unknown")
            source = metadata.get("source", "transcript")
            passages.append(f"[Lecture {date}, {source}]\n{doc}")

        context = "\n\n---\n\n".join(passages)
        console.print(f"[dim]RAG: retrieved {len(passages)} passages from previous lectures[/dim]")
        return context

    def add_from_pdf(self, pdf_path: Path, course_name: str) -> int:
        """
        Extract text from a PDF and add it to the RAG database.
        Returns the number of chunks added.
        """
        try:
            import pdfplumber
        except ImportError:
            console.print("[yellow]⚠ pdfplumber not installed — cannot index PDFs[/yellow]")
            return 0

        text_parts = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
        except Exception as e:
            console.print(f"[yellow]⚠ Failed to read PDF {pdf_path}: {e}[/yellow]")
            return 0

        if not text_parts:
            console.print(f"[dim]RAG: no text extracted from {pdf_path.name}[/dim]")
            return 0

        full_text = "\n\n".join(text_parts)

        # Extract date from filename (expected format: DD-MM-YYYY_CourseName.pdf)
        lecture_date = "unknown"
        name = pdf_path.stem
        parts = name.split("_", 1)
        if parts and len(parts[0]) == 10:
            lecture_date = parts[0]

        return self.add_lecture(
            transcript=full_text,
            course_name=course_name,
            lecture_date=lecture_date,
            source="pdf",
        )

    def course_exists(self, course_name: str) -> bool:
        """Check if any documents exist for this course."""
        try:
            collection = self._get_collection(course_name)
            return collection.count() > 0
        except Exception:
            return False
