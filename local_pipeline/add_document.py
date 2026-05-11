import sys
import json
import time
import numpy as np
import faiss
import fitz
import tiktoken
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError, BadRequestError
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Paramètres
EMBEDDING_MODEL = "text-embedding-3-large"
MAX_TOKENS = 800
OVERLAP = 150
RATE_LIMIT_SLEEP = 1.0
CONNECTION_ERROR_SLEEP = 5.0

# Fichiers
VECTOR_DIR = Path("vector_store")
INDEX_FILE = VECTOR_DIR / "faiss.index"
METADATA_FILE = VECTOR_DIR / "metadata.json"
CORPUS_FILE = Path("extracted_text/corpus.txt")

enc = tiktoken.get_encoding("cl100k_base")


def extract_text(pdf_path: Path):
    """Extrait le texte page par page."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "file": pdf_path.name,
                "page": i,
                "text": text
            })
    print(f"  {len(pages)} pages extraites")
    return pages


def chunk_pages(pages):
    """Découpe les pages en chunks avec overlap."""
    chunks = []
    for page in pages:
        tokens = enc.encode(page["text"])
        start = 0
        while start < len(tokens):
            end = start + MAX_TOKENS
            chunk_text = enc.decode(tokens[start:end])
            chunks.append({
                "source": page["file"],
                "page": page["page"],
                "text": chunk_text,
                "tokens": len(enc.encode(chunk_text))
            })
            start = end - OVERLAP
            if start < 0:
                start = 0
    print(f"  {len(chunks)} chunks générés")
    return chunks


def build_embeddings(chunks):
    """Calcule les embeddings avec gestion des erreurs."""
    vectors = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"].strip() or "vide"
        while True:
            try:
                r = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=text
                )
                vectors.append(r.data[0].embedding)
                break
            except RateLimitError:
                print(f"  Rate limit au chunk {i+1} — pause {RATE_LIMIT_SLEEP}s…")
                time.sleep(RATE_LIMIT_SLEEP)
            except APIConnectionError:
                print(f"  Erreur réseau au chunk {i+1} — retry dans {CONNECTION_ERROR_SLEEP}s…")
                time.sleep(CONNECTION_ERROR_SLEEP)
            except BadRequestError as e:
                print(f"  Chunk {i+1} invalide, ignoré : {e}")
                vectors.append([0.0] * 3072)
                break

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  {i+1} / {len(chunks)} embeddings calculés")

    return np.array(vectors, dtype="float32")


def update_corpus(pages):
    """Ajoute le texte du nouveau document à corpus.txt."""
    with open(CORPUS_FILE, "a", encoding="utf-8") as f:
        for page in pages:
            f.write(f"[{page['file']} — page {page['page']}]\n")
            f.write(page["text"])
            f.write("\n\n")
    print(f"  corpus.txt mis à jour")


def update_index(vectors, chunks):
    """Fusionne les nouveaux vecteurs dans l'index FAISS existant."""
    index = faiss.read_index(str(INDEX_FILE))
    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    index.add(vectors)
    meta += [{"source": c["source"], "page": c["page"], "tokens": c["tokens"]} for c in chunks]

    faiss.write_index(index, str(INDEX_FILE))
    METADATA_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Index FAISS et metadata mis à jour")


def main():
    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        pdf_path = Path(input("Chemin du PDF à ajouter : ").strip())

    if not pdf_path.exists():
        print(f"Erreur : fichier introuvable → {pdf_path}")
        return

    print(f"\n→ Traitement de : {pdf_path.name}")

    print("\n[1/4] Extraction du texte…")
    pages = extract_text(pdf_path)

    print("\n[2/4] Découpage en chunks…")
    chunks = chunk_pages(pages)

    print("\n[3/4] Calcul des embeddings…")
    vectors = build_embeddings(chunks)

    print("\n[4/4] Mise à jour de la base…")
    update_corpus(pages)
    update_index(vectors, chunks)

    print(f"\n✅ {pdf_path.name} ajouté avec succès ({len(chunks)} chunks)")


if __name__ == "__main__":
    main()
