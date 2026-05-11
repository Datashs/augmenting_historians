"""
add_document_local.py
=====================
Ajoute un ou plusieurs PDFs à l'index FAISS LOCAL existant (vector_store_local/).
100 % local — aucune API externe, aucune clé requise.

Cohérent avec :
    03_build_embeddings_local.py  → construit l'index initial
    rag_config_local.py           → config partagée (modèle, chemins, etc.)

Usage :
    # Un seul fichier
    python add_document_local.py mon_fichier.pdf

    # Tous les PDFs d'un dossier (non récursif)
    python add_document_local.py mon_dossier/

    # Tous les PDFs d'un dossier et ses sous-dossiers
    python add_document_local.py mon_dossier/ --recursive

    # Re-indexer même les fichiers déjà présents
    python add_document_local.py mon_dossier/ --force

    # Mode interactif (demande le chemin)
    python add_document_local.py
"""

import sys
import json
import argparse
import numpy as np
import faiss
import fitz
import tiktoken
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Config partagée avec le reste du pipeline local
from rag_config_local import (
    VECTOR_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
)

# =============================================================================
# PARAMÈTRES
# =============================================================================

MAX_TOKENS = 800
OVERLAP    = 150

VECTOR_PATH     = Path(VECTOR_DIR)
INDEX_FILE      = VECTOR_PATH / "faiss.index"
METADATA_FILE   = VECTOR_PATH / "metadata.json"
EMBEDDINGS_FILE = VECTOR_PATH / "embeddings.npy"
CORPUS_FILE     = Path("extracted_text/corpus.txt")

enc = tiktoken.get_encoding("cl100k_base")


# =============================================================================
# RÉSOLUTION DE LA CIBLE (fichier ou dossier)
# =============================================================================

def resolve_pdfs(target: Path, recursive: bool) -> list[Path]:
    """
    Retourne la liste des PDFs à traiter selon que la cible
    est un fichier unique ou un dossier.

    Args:
        target    : Chemin vers un PDF ou un dossier.
        recursive : Si True, parcourt les sous-dossiers.

    Returns:
        Liste triée de Path vers des fichiers .pdf.

    Raises:
        FileNotFoundError : Si la cible n'existe pas.
        ValueError        : Si aucun PDF n'est trouvé dans le dossier.
    """
    if not target.exists():
        raise FileNotFoundError(f"Cible introuvable : {target.resolve()}")

    if target.is_file():
        if target.suffix.lower() != ".pdf":
            raise ValueError(f"Le fichier n'est pas un PDF : {target}")
        return [target]

    # Dossier
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs    = sorted(target.glob(pattern))

    if not pdfs:
        suffix = " (et sous-dossiers)" if recursive else ""
        raise ValueError(f"Aucun PDF trouvé dans {target.resolve()}{suffix}")

    return pdfs


def already_indexed(pdf_name: str, metadata: list[dict]) -> bool:
    """Vérifie si un PDF est déjà présent dans les métadonnées."""
    return any(entry["source"] == pdf_name for entry in metadata)


# =============================================================================
# EXTRACTION / CHUNKING / EMBEDDINGS
# =============================================================================

def extract_text(pdf_path: Path) -> list[dict]:
    """Extrait le texte page par page avec PyMuPDF."""
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        if text:
            pages.append({"file": pdf_path.name, "page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Découpe les pages en chunks avec overlap."""
    chunks = []
    for page in pages:
        tokens = enc.encode(page["text"])
        start  = 0
        while start < len(tokens):
            end        = min(start + MAX_TOKENS, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append({
                "source": page["file"],
                "page":   page["page"],
                "text":   chunk_text,
                "tokens": len(enc.encode(chunk_text)),
            })
            if end == len(tokens):
                break
            start = end - OVERLAP
    return chunks


def clean_texts(texts: list[str]) -> list[str]:
    """Nettoie les textes avant vectorisation."""
    cleaned = []
    for text in texts:
        if not text or not text.strip():
            cleaned.append("vide")
        else:
            cleaned.append(text.replace("\x00", "").strip() or "vide")
    return cleaned


def build_embeddings_local(
    model: SentenceTransformer,
    chunks: list[dict],
    label: str = "",
) -> np.ndarray:
    """Calcule les embeddings par lots via sentence-transformers."""
    texts   = clean_texts([c["text"] for c in chunks])
    total   = len(texts)
    vectors = []

    for batch_start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch_end     = min(batch_start + EMBEDDING_BATCH_SIZE, total)
        batch_vectors = model.encode(
            texts[batch_start:batch_end],
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors.extend(batch_vectors)
        print(f"    {label}{batch_end} / {total} embeddings")

    return np.array(vectors, dtype="float32")


# =============================================================================
# GESTION DE L'INDEX
# =============================================================================

def load_index_and_meta() -> tuple[faiss.Index, list[dict]]:
    """Charge l'index FAISS et les métadonnées existants."""
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Index FAISS introuvable : {INDEX_FILE.resolve()}\n"
            "Lancez d'abord 03_build_embeddings_local.py pour créer l'index initial."
        )
    index    = faiss.read_index(str(INDEX_FILE))
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    return index, metadata


def save_all(index: faiss.Index, metadata: list[dict], new_vectors: np.ndarray) -> None:
    """Sauvegarde l'index FAISS, les métadonnées et embeddings.npy."""
    faiss.write_index(index, str(INDEX_FILE))
    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if EMBEDDINGS_FILE.exists():
        existing = np.load(EMBEDDINGS_FILE)
        updated  = np.vstack([existing, new_vectors])
    else:
        updated = new_vectors
    np.save(EMBEDDINGS_FILE, updated)


def update_corpus(pages: list[dict]) -> None:
    """Ajoute le texte extrait à corpus.txt."""
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_FILE, "a", encoding="utf-8") as f:
        for page in pages:
            f.write(f"[{page['file']} — page {page['page']}]\n")
            f.write(page["text"])
            f.write("\n\n")


# =============================================================================
# TRAITEMENT D'UN SEUL PDF
# =============================================================================

def process_pdf(
    pdf_path: Path,
    model: SentenceTransformer,
    index: faiss.Index,
    metadata: list[dict],
) -> tuple[int, np.ndarray]:
    """
    Traite un PDF : extraction → chunking → embeddings → mise à jour en mémoire.
    La sauvegarde disque est différée (faite une seule fois à la fin).

    Returns:
        (nb_chunks, vecteurs numpy) — (0, array vide) si rien extrait.
    """
    print(f"  → Extraction du texte…")
    pages = extract_text(pdf_path)
    if not pages:
        print("  ⚠ Aucun texte extrait (PDF image/scanné ?). Fichier ignoré.")
        return 0, np.empty((0, index.d), dtype="float32")

    print(f"  → {len(pages)} pages | découpage en chunks…")
    chunks = chunk_pages(pages)
    print(f"  → {len(chunks)} chunks | calcul des embeddings…")
    vectors = build_embeddings_local(model, chunks)

    # Vérification de cohérence de dimension
    if vectors.shape[1] != index.d:
        raise ValueError(
            f"Incompatibilité de dimension : index={index.d}d, "
            f"nouveaux vecteurs={vectors.shape[1]}d.\n"
            "Vérifiez que EMBEDDING_MODEL dans rag_config_local.py n'a pas changé."
        )

    # Mise à jour en mémoire (pas encore sur disque)
    index.add(vectors)
    metadata += [
        {"source": c["source"], "page": c["page"], "tokens": c["tokens"]}
        for c in chunks
    ]
    update_corpus(pages)

    return len(chunks), vectors


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ajoute un PDF ou tous les PDFs d'un dossier à l'index FAISS local.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python add_document_local.py article.pdf
  python add_document_local.py ./nouveaux_docs/
  python add_document_local.py ./archives/ --recursive
  python add_document_local.py ./docs/ --force
        """,
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Chemin vers un fichier PDF ou un dossier contenant des PDFs.",
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Parcourt les sous-dossiers (uniquement si target est un dossier).",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-indexe les fichiers déjà présents dans la base.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Résolution de la cible
    if args.target:
        target = Path(args.target)
    else:
        raw    = input("Chemin du PDF ou du dossier à indexer : ").strip()
        target = Path(raw)

    try:
        pdfs = resolve_pdfs(target, recursive=args.recursive)
    except (FileNotFoundError, ValueError) as e:
        print(f"Erreur : {e}")
        sys.exit(1)

    mode = "dossier" if target.is_dir() else "fichier"
    print(f"\n{len(pdfs)} PDF(s) détecté(s) ({mode})")
    for p in pdfs:
        print(f"  • {p.name}")

    # Chargement de l'index existant
    print(f"\nChargement de l'index FAISS local…")
    try:
        index, metadata = load_index_and_meta()
    except FileNotFoundError as e:
        print(f"Erreur : {e}")
        sys.exit(1)
    print(f"  {index.ntotal} vecteurs existants | {len(metadata)} entrées")

    # Chargement du modèle (une seule fois pour tous les fichiers)
    print(f"\nChargement du modèle : {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Modèle chargé (dim={model.get_sentence_embedding_dimension()})")

    # Traitement de chaque PDF
    total_chunks    = 0
    all_new_vectors = []
    skipped         = []
    failed          = []

    for i, pdf_path in enumerate(pdfs, start=1):
        print(f"\n[{i}/{len(pdfs)}] {pdf_path.name}")

        # Vérification doublon
        if not args.force and already_indexed(pdf_path.name, metadata):
            print(f"  ⏭  Déjà indexé — ignoré (utilisez --force pour réindexer)")
            skipped.append(pdf_path.name)
            continue

        try:
            nb_chunks, vectors = process_pdf(pdf_path, model, index, metadata)
            if nb_chunks > 0:
                total_chunks += nb_chunks
                all_new_vectors.append(vectors)
                print(f"  ✓ {nb_chunks} chunks ajoutés")
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
            failed.append(pdf_path.name)

    # Sauvegarde finale (une seule écriture disque)
    if all_new_vectors:
        merged = np.vstack(all_new_vectors)
        print(f"\nSauvegarde ({index.ntotal} vecteurs au total)…")
        save_all(index, metadata, merged)
        print("  ✓ faiss.index, metadata.json, embeddings.npy mis à jour")
    else:
        print("\nAucun nouveau vecteur à sauvegarder.")

    # Résumé final
    print("\n" + "=" * 52)
    print(f"  PDFs traités           : {len(pdfs) - len(skipped) - len(failed)}")
    print(f"  Chunks ajoutés         : {total_chunks}")
    if skipped:
        print(f"  Ignorés (déjà indexés) : {len(skipped)}")
        for name in skipped:
            print(f"    - {name}")
    if failed:
        print(f"  Échecs                 : {len(failed)}")
        for name in failed:
            print(f"    - {name}")
    print("=" * 52)
    if total_chunks > 0:
        print("\n  Étape suivante : python 04_rag_query_local.py")


if __name__ == "__main__":
    main()
