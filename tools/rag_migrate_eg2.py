#!/usr/bin/env python3
"""
rag_migrate_eg2.py — Copia un database RAG (ChromaDB) ricalcolando gli embedding con EmbeddingGemma 2.

Chroma conserva il testo dei chunk e i metadati: non serve rileggere trascrizioni e PDF. Il
database di origine non viene toccato (resta come backup); quello nuovo ha le stesse collezioni,
gli stessi id e metadati, e "embedder": "embeddinggemma-2" nei metadati della collezione.

  .venv/bin/python tools/rag_migrate_eg2.py                       # output/rag → output/rag_eg2
  .venv/bin/python tools/rag_migrate_eg2.py --src A --dst B --force   # --force: ricrea le collezioni già presenti
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="output/rag")
    ap.add_argument("--dst", default="output/rag_eg2")
    ap.add_argument("--force", action="store_true", help="ricrea le collezioni già presenti in dst")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()

    import chromadb
    from src.slides import eg2
    if not eg2.available():
        print("torch / sentence-transformers non installati")
        return 1
    if Path(a.src).resolve() == Path(a.dst).resolve():
        print("src e dst devono essere diversi")
        return 1
    src = chromadb.PersistentClient(path=a.src)
    dst = chromadb.PersistentClient(path=a.dst)
    have = {c.name for c in dst.list_collections()}
    t0 = time.time()
    for col in src.list_collections():
        if col.name in have:
            if not a.force:
                print(f"{col.name}: già presente in {a.dst}, saltata (--force per ricrearla)")
                continue
            dst.delete_collection(col.name)
        data = col.get(include=["documents", "metadatas"])
        new = dst.create_collection(col.name, metadata={**(col.metadata or {}), "embedder": "embeddinggemma-2"})
        n = len(data["ids"])
        for i in range(0, n, a.batch):
            docs = data["documents"][i:i + a.batch]
            new.add(ids=data["ids"][i:i + a.batch], documents=docs,
                    metadatas=data["metadatas"][i:i + a.batch],
                    embeddings=eg2.encode_texts(docs, prompt_name="Document").tolist())
        print(f"{col.name}: {n} chunk ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
