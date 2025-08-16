#!/usr/bin/env python3
"""
Enhanced Scalable RAG CLI
-------------------------
Features:
- Deduplication by SHA256 + embedding similarity
- Optional summarization for very long docs
- Chunking with overlap
- Metadata tagging: source, type, section, chunk_index
- Hallucination mitigation in LLM prompts
- Together.ai default backend: mistral-7b-instruct
"""

import argparse, hashlib, json, os, shutil
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

# ---------------- Constants ----------------
STORE_DIR = Path("./rag_store")
COLLECTION_NAME = "documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
INGESTED_JSON = STORE_DIR / "ingested_files.json"

DEFAULT_CHUNK_SIZE = 300
DEFAULT_OVERLAP = 50
SUMMARIZE_THRESHOLD = 2000  # words
SIMILARITY_THRESHOLD = 0.95  # embedding similarity dedup

# ---------------- Chroma ----------------
client = chromadb.PersistentClient(path=str(STORE_DIR), settings=Settings(allow_reset=True))
collection = client.get_or_create_collection(COLLECTION_NAME)

# ---------------- Embeddings ----------------
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- Already-ingested files ----------------
if INGESTED_JSON.exists():
    seen_hashes = set(json.loads(INGESTED_JSON.read_text()))
else:
    seen_hashes = set()

# ---------------- Loaders ----------------
def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf(path: Path) -> str:
    if not PdfReader:
        raise RuntimeError("pypdf not installed")
    text = ""
    reader = PdfReader(str(path))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_docx(path: Path) -> str:
    if not docx:
        raise RuntimeError("python-docx not installed")
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)

def convert_doc_to_docx(doc_path: Path) -> Path:
    """
    Convert a .doc file to .docx using Word COM on Windows.
    Returns the Path to the new .docx file.
    """
    if doc_path.suffix.lower() != ".doc":
        raise ValueError(f"{doc_path} is not a .doc file")

    try:
        import win32com.client
    except ImportError:
        raise RuntimeError("pywin32 is required for .doc conversion. Install with `pip install pywin32`")

    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False

    doc = word.Documents.Open(str(doc_path))
    docx_path = doc_path.with_suffix(".docx")
    doc.SaveAs(str(docx_path), FileFormat=16)  # 16 = wdFormatDocumentDefault (.docx)
    doc.Close()
    word.Quit()

    print(f"Converted {doc_path} -> {docx_path}")
    return docx_path

# Example usage
folder = Path("docs/")
for file in folder.glob("*.doc"):
    convert_doc_to_docx(file)

# ---------------- Chunking ----------------
def chunk_text(text: str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

# ---------------- Deduplication ----------------
def filter_similar(chunks: List[str], threshold=SIMILARITY_THRESHOLD) -> List[str]:
    embeddings = embedder.encode(chunks)
    keep = []
    for i, chunk in enumerate(chunks):
        if not keep:
            keep.append(i)
            continue
        sims = cosine_similarity([embeddings[i]], embeddings[keep])
        if np.max(sims) < threshold:
            keep.append(i)
    return [chunks[i] for i in keep]

# ---------------- Summarization ----------------
def summarize_long_text(text: str, backend="together", model="mistral-7b-instruct") -> str:
    if len(text.split()) < SUMMARIZE_THRESHOLD:
        return text
    prompt = f"Summarize this text for RAG retrieval:\n\n{text}"
    return call_together(model, prompt)

# ---------------- Ingestion ----------------
def ingest_path(path: Path, doc_type="general"):
    files = []
    if path.is_dir():
        for ext in ("*.txt", "*.pdf", "*.docx", "*.doc"):
            files.extend(path.rglob(ext))
    else:
        files.append(path)

    for file in files:
        fhash = hashlib.sha256(file.read_bytes()).hexdigest()
        if fhash in seen_hashes:
            print(f"Skipping {file} (already ingested)")
            continue

        print(f"Ingesting {file}")
        # Load file

        if file.suffix.lower() == ".doc":
            new_file = convert_doc_to_docx(file)
            file.unlink()
            file = new_file
            text = load_docx(file)
        elif file.suffix.lower() == ".txt":
            text = load_txt(file)
        elif file.suffix.lower() == ".pdf":
            text = load_pdf(file)
        elif file.suffix.lower() == ".docx":
            text = load_docx(file)
        else:
            print(f"Unsupported file: {file}")
            continue

        # Optional: summarize
        # text = summarize_long_text(text)

        # Chunk + deduplicate
        chunks = chunk_text(text)
        chunks = filter_similar(chunks)

        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{fhash}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": str(file), "type": doc_type, "section": f"chunk_{i}", "chunk_index": i} for i in range(len(chunks))]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

        # Update ingested hashes
        seen_hashes.add(fhash)
        INGESTED_JSON.write_text(json.dumps(list(seen_hashes)))

    print("✅ Ingestion complete.")

# ---------------- Retrieval ----------------
def retrieve(query: str, k: int = 3, metadata_filter: dict = None):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=k, where=metadata_filter)
    docs = [d for sub in results["documents"] for d in sub]
    return docs

# ---------------- LLM Backends ----------------
def call_together(model: str, prompt: str) -> str:
    api_key = "6c86d1e21783273b45c286dad76384487fff5a99f0cfe339de63f0ecd8bb765c"
    if not api_key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    import openai
    client = openai.OpenAI(base_url="https://api.together.xyz/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content

def call_ollama(model: str, prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    r = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False})
    r.raise_for_status()
    return r.json().get("response", "")

# ---------------- Query ----------------
def query_rag(query: str, k: int, backend: str, model: str, metadata_filter: dict = None):
    docs = retrieve(query, k=k, metadata_filter=metadata_filter)
    context = "\n".join(docs)
    prompt = f"""Use the following context to answer the question.
Only answer if the information exists in the context. Otherwise, say "I don't know."

Context:
{context}

Question:
{query}"""
    if backend == "together":
        answer = call_together(model, prompt)
    elif backend == "ollama":
        answer = call_ollama(model, prompt)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    print("🤖", answer)

# ---------------- Utilities ----------------
def stats():
    count = collection.count()
    print(f"📦 {count} chunks in vector store.")

def wipe():
    print("⚠ Wiping vector store...")
    shutil.rmtree(STORE_DIR, ignore_errors=True)
    print("✅ Store deleted.")
    global seen_hashes
    seen_hashes = set()
    if INGESTED_JSON.exists():
        INGESTED_JSON.unlink()

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser(description="Enhanced Scalable RAG CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="Ingest file or folder")
    p_ingest.add_argument("path", type=str, help="Path to file or folder")
    p_ingest.add_argument("--type", type=str, default="general", help="Document type (interview, essay, etc.)")

    p_query = sub.add_parser("query", help="Query the RAG")
    p_query.add_argument("query", type=str, help="Your question")
    p_query.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    p_query.add_argument("--backend", type=str, default="together", help="LLM backend")
    p_query.add_argument("--model", type=str, default="mistral-7b-instruct", help="Model to use")
    p_query.add_argument("--type", type=str, default=None, help="Optional doc type filter")

    sub.add_parser("stats", help="Show store stats")
    sub.add_parser("wipe", help="Delete the store")

    args = parser.parse_args()

    if args.cmd == "ingest":
        ingest_path(Path(args.path), doc_type=args.type)
    elif args.cmd == "query":
        metadata_filter = {"type": args.type} if args.type else None
        query_rag(args.query, args.k, args.backend, args.model, metadata_filter)
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "wipe":
        wipe()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
