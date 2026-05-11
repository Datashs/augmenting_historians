"""
tool_semantic_search.py
=======================
Outil de diagnostic du pipeline RAG documentaire.

Rôle : Recherche et affiche les chunks les plus proches d'une question
dans l'index FAISS, SANS appel au LLM. Utile pour inspecter la qualité
de l'index et vérifier que les bons passages sont retrouvés avant de
lancer 04_rag_query.py.

Cas d'usage typiques :
    - Vérifier que le chunking et les embeddings sont cohérents.
    - Comprendre pourquoi 04_rag_query.py retourne une mauvaise réponse
      (le problème vient-il de la recherche ou du LLM ?).
    - Explorer rapidement le corpus sans coût LLM.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → réponse synthétique via LLM
    ──
    tool_semantic_search.py  diagnostic de recherche sans LLM           (ce script)
    tool_add_document.py     ajout incrémental d'un PDF au corpus

Structure de fichiers attendue :
    projet/
    ├── .env                         ← OPENAI_API_KEY=sk-...
    ├── tool_semantic_search.py
    └── vector_store/
        ├── faiss.index              ← requis
        └── metadata.json            ← requis
"""

# =============================================================================
# PARAMÈTRES — ajustez selon votre contexte
# =============================================================================

# --- Chemins ---
VECTOR_DIR = "vector_store"

# --- Modèle d'embeddings ---
# Doit être identique au modèle utilisé dans build_embeddings.py.
# Un modèle différent produirait des vecteurs incompatibles avec l'index.
EMBEDDING_MODEL = "text-embedding-3-large"

# --- Recherche ---
# Nombre de chunks retournés par FAISS.
# Valeurs typiques : 3–10 pour l'exploration, cohérent avec TOP_K de rag_query.py.
TOP_K = 5

# --- Affichage ---
# Afficher le score de distance FAISS pour chaque résultat.
# Distance L2 : plus elle est faible, plus le chunk est proche de la question.
# Utile pour détecter des résultats peu pertinents (distance très élevée).
SHOW_DISTANCE = True

# Nombre maximum de caractères du texte du chunk à afficher.
# None pour afficher le chunk complet.
PREVIEW_LENGTH = 300

# =============================================================================
# IMPORTS
# =============================================================================

import json
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# =============================================================================
# INITIALISATION
# =============================================================================

load_dotenv()
client = OpenAI()  # Lit automatiquement OPENAI_API_KEY depuis .env

vector_dir    = Path(VECTOR_DIR)
index_file    = vector_dir / "faiss.index"
metadata_file = vector_dir / "metadata.json"

# =============================================================================
# FONCTIONS
# =============================================================================

def load_vector_store() -> tuple[faiss.Index, list[dict]]:
    """
    Charge l'index FAISS et les métadonnées depuis le dossier vector_store.

    Returns:
        index    : Index FAISS prêt pour la recherche.
        metadata : Liste de dicts {source, page, tokens} indexée comme l'index FAISS.

    Raises:
        FileNotFoundError : Si faiss.index ou metadata.json sont absents.
    """
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path.resolve()}\n"
                "Lancez d'abord build_embeddings.py pour générer le vector store."
            )

    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


def embed_query(query: str) -> np.ndarray:
    """
    Encode la question en vecteur via l'API OpenAI.

    Le modèle utilisé doit être identique à celui de build_embeddings.py,
    sans quoi les distances calculées par FAISS n'auraient aucun sens.

    Args:
        query: Question posée par l'utilisateur.

    Returns:
        Vecteur float32 de forme (1, EMBEDDING_DIM) prêt pour index.search().
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    vector = np.array(response.data[0].embedding, dtype="float32")
    return vector.reshape(1, -1)


def display_results(indices: np.ndarray, distances: np.ndarray, metadata: list[dict]) -> None:
    """
    Affiche les chunks retrouvés avec leurs métadonnées et score de distance.

    La distance L2 indique la proximité sémantique entre la question et le chunk :
    une valeur faible signifie une forte similarité. Il n'y a pas de seuil universel,
    mais des valeurs très élevées (ex. > 1.5 pour text-embedding-3-large) indiquent
    un résultat probablement hors sujet.

    Args:
        indices   : Indices FAISS des chunks retrouvés, forme (1, TOP_K).
        distances : Distances L2 associées, forme (1, TOP_K).
        metadata  : Liste complète des métadonnées du vector store.
    """
    print(f"\n--- {TOP_K} passages les plus proches ---\n")

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        meta = metadata[idx]

        header = f"[{rank}] {meta['source']} — page {meta['page']}  ({meta['tokens']} tokens)"
        if SHOW_DISTANCE:
            header += f"  |  distance L2 : {dist:.4f}"

        print(header)

        # Aperçu du texte si disponible dans les métadonnées
        if "text" in meta:
            preview = meta["text"]
            if PREVIEW_LENGTH and len(preview) > PREVIEW_LENGTH:
                preview = preview[:PREVIEW_LENGTH] + "…"
            print(preview)

        print("-" * 60)

    print("\n(Consultez le texte complet dans corpus.txt)\n")


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    index, metadata = load_vector_store()
    print(f"Index chargé : {index.ntotal} vecteurs | modèle : {EMBEDDING_MODEL}")

    query = input("\nVotre question : ").strip()
    if not query:
        print("Question vide, abandon.")
        return

    q_vector = embed_query(query)
    distances, indices = index.search(q_vector, TOP_K)

    display_results(indices, distances, metadata)


if __name__ == "__main__":
    main()
