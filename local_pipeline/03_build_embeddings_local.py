"""
03_build_embeddings_local.py
============================
Étape 3 du pipeline RAG documentaire — VERSION 100 % LOCALE.

Rôle : Génère les embeddings vectoriels de chaque chunk de texte via
sentence-transformers (aucune API externe), puis construit un index FAISS
pour la recherche par similarité.

Pipeline :
    01_extract_text.py              → extracted_text/corpus.txt
    02_chunk_corpus.py              → extracted_text/chunks.json
    03_build_embeddings_local.py    → vector_store_local/{embeddings.npy,
                                       faiss.index, metadata.json}    (ce script)
    04_rag_query_local.py           → réponse synthétique via Ollama
    ──

Différences avec la version OpenAI (03_build_embeddings.py) :
    - Pas de clé API, pas de quota, pas de coût par token.
    - Vectorisation en local via sentence-transformers.
    - Dossier de sortie distinct : vector_store_local/ (vecteurs de dimension 768
      contre 3072 pour text-embedding-3-large — les deux index sont incompatibles).
    - Traitement par lots (batch) pour optimiser l'usage du CPU/GPU.

Structure de fichiers attendue :
    projet/
    ├── 00_config_local.py
    ├── 03_build_embeddings_local.py
    ├── extracted_text/
    │   └── chunks.json              ← produit par 02_chunk_corpus.py (requis)
    └── vector_store_local/          ← créé automatiquement par ce script
        ├── embeddings.npy
        ├── faiss.index
        └── metadata.json

Prérequis :
    pip install sentence-transformers faiss-cpu numpy
    # Le modèle (~1,1 Go) est téléchargé automatiquement au premier lancement
    # et mis en cache dans ~/.cache/huggingface/
"""

# =============================================================================
# PARAMÈTRES — surchargent les valeurs de 00_config_local.py si besoin
# =============================================================================

# Importer la config centrale (chemins, modèles, dimensions)
from rag_config_local import (
    CHUNKS_FILE,
    VECTOR_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_TYPE,
)

# --- Sauvegarde intermédiaire ---
# Protection contre les interruptions : sauvegarde tous les N chunks.
# Utile pour les très grands corpus (> 10 000 chunks).
SAVE_EVERY_N_CHUNKS = 500

# Affichage de la progression tous les N lots
LOG_EVERY_N_BATCHES = 5

# --- FAISS : paramètres avancés (ignorés si FAISS_INDEX_TYPE = "flat") ---
# Nombre de clusters pour IndexIVFFlat
IVF_NLIST = 100
# Nombre de voisins pour IndexHNSWFlat (valeurs typiques : 16–64)
HNSW_M = 32

# =============================================================================
# IMPORTS
# =============================================================================

import json
import numpy as np
import faiss
from pathlib import Path

# sentence-transformers : pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# =============================================================================
# INITIALISATION
# =============================================================================

vector_path      = Path(VECTOR_DIR)
chunks_path      = Path(CHUNKS_FILE)
embeddings_file  = vector_path / "embeddings.npy"
index_file       = vector_path / "faiss.index"
metadata_file    = vector_path / "metadata.json"

# =============================================================================
# FONCTIONS
# =============================================================================

def load_chunks() -> tuple[list[str], list[dict]]:
    """
    Charge le fichier chunks.json produit par 02_chunk_corpus.py.

    Format attendu : liste de dicts avec les clés
        - "text"   : contenu textuel du chunk
        - "source" : nom du fichier source (ex : "manuscrit.pdf")
        - "page"   : numéro de page
        - "tokens" : nombre de tokens estimé

    Returns:
        texts    : liste des textes (un par chunk)
        metadata : liste des métadonnées associées

    Raises:
        FileNotFoundError : si chunks.json est absent.
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chunks_path.resolve()}\n"
            "Lancez d'abord 02_chunk_corpus.py pour générer ce fichier."
        )

    chunks   = json.loads(chunks_path.read_text(encoding="utf-8"))
    texts    = [c["text"] for c in chunks]
    metadata = [
        {"source": c["source"], "page": c["page"], "tokens": c["tokens"]}
        for c in chunks
    ]

    print(f"{len(texts)} chunks chargés depuis {chunks_path}")
    return texts, metadata


def load_existing_embeddings() -> tuple[list, int]:
    """
    Charge les embeddings déjà calculés pour reprendre un traitement interrompu.

    Si embeddings.npy existe, le script reprend depuis le dernier chunk traité
    sans tout recalculer — utile pour les très grands corpus.

    Returns:
        embeddings  : liste de vecteurs numpy déjà calculés (vide si aucun)
        start_index : index du prochain chunk à traiter
    """
    if embeddings_file.exists():
        embeddings  = list(np.load(embeddings_file))
        start_index = len(embeddings)
        print(f"Reprise détectée : {start_index} embeddings existants.")
        return embeddings, start_index

    return [], 0


def clean_texts(texts: list[str]) -> list[str]:
    """
    Nettoie une liste de textes avant vectorisation.

    - Remplace les textes vides par la chaîne sentinelle "vide"
      (sentence-transformers gère les chaînes vides, mais produit
      un vecteur non représentatif).
    - Supprime les caractères nuls (\x00) potentiellement issus de l'OCR.

    Args:
        texts: Liste brute des textes de chunks.

    Returns:
        Liste nettoyée, même longueur que l'entrée.
    """
    cleaned = []
    for text in texts:
        if not text or not text.strip():
            cleaned.append("vide")
        else:
            text = text.replace("\x00", "")
            cleaned.append(text.strip() or "vide")
    return cleaned


def embed_texts_in_batches(
    model: SentenceTransformer,
    texts: list[str],
    start_index: int,
    existing_embeddings: list,
) -> np.ndarray:
    """
    Vectorise les textes par lots (batch) et gère la reprise.

    Traitement par lots plutôt que chunk par chunk :
        - Réduit la surcharge Python par appel.
        - Exploite mieux le parallélisme CPU (et GPU si disponible).
        - Sur Mac M2, sentence-transformers utilise MPS (Metal Performance
          Shaders) automatiquement si disponible.

    Scores de similarité cosinus produits par ce modèle :
        > 0.85  : passages très similaires (quasi-identiques)
        0.60–0.85 : passages sur le même sujet
        0.40–0.60 : relation thématique lointaine
        < 0.40  : probablement sans rapport
        Ces valeurs sont indicatives ; elles varient selon le corpus.

    Args:
        model             : Modèle SentenceTransformer chargé.
        texts             : Textes à vectoriser (déjà nettoyés).
        start_index       : Index du premier chunk à traiter (reprise).
        existing_embeddings: Vecteurs déjà calculés lors d'une session précédente.

    Returns:
        Matrice numpy float32 de forme (n_chunks_total, EMBEDDING_DIM).
    """
    remaining_texts = texts[start_index:]
    total_remaining = len(remaining_texts)
    embeddings      = list(existing_embeddings)

    if total_remaining == 0:
        print("Tous les embeddings sont déjà calculés.")
        return np.array(embeddings, dtype="float32")

    print(
        f"Vectorisation de {total_remaining} chunks "
        f"(lots de {EMBEDDING_BATCH_SIZE})…"
    )

    for batch_start in range(0, total_remaining, EMBEDDING_BATCH_SIZE):
        batch_end  = min(batch_start + EMBEDDING_BATCH_SIZE, total_remaining)
        batch      = remaining_texts[batch_start:batch_end]

        # encode() retourne un tableau numpy (n_batch, dim)
        # show_progress_bar=False car on gère nous-mêmes l'affichage
        batch_vectors = model.encode(
            batch,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,   # L2 brut pour FAISS IndexFlatL2
        )

        embeddings.extend(batch_vectors)

        # Progression
        batch_num = batch_start // EMBEDDING_BATCH_SIZE + 1
        if batch_num % LOG_EVERY_N_BATCHES == 0 or batch_end == total_remaining:
            done = start_index + batch_end
            print(f"  {done} / {start_index + total_remaining} chunks")

        # Sauvegarde intermédiaire
        global_done = start_index + batch_end
        if global_done % SAVE_EVERY_N_CHUNKS < EMBEDDING_BATCH_SIZE:
            np.save(embeddings_file, np.array(embeddings, dtype="float32"))
            print(f"  Sauvegarde intermédiaire ({global_done} chunks)")

    return np.array(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Construit un index FAISS à partir d'une matrice d'embeddings.

    Le type d'index est contrôlé par FAISS_INDEX_TYPE (dans 00_config_local.py) :
    - "flat"  : IndexFlatL2 — recherche exacte, recommandé jusqu'à ~50 000 chunks.
                Aucune perte de précision, pas d'entraînement.
    - "ivf"   : IndexIVFFlat — clustering. Rapide sur grands volumes (> 50 000),
                mais perd ~1–5 % de précision. Nécessite un entraînement.
    - "hnsw"  : IndexHNSWFlat — graphe de proximité. Très rapide et précis,
                consomme plus de RAM.

    Pour un corpus historique de taille typique (quelques milliers de chunks),
    "flat" est largement suffisant.

    Args:
        embeddings: Matrice float32 de forme (n_chunks, EMBEDDING_DIM).

    Returns:
        Index FAISS prêt pour la recherche.

    Raises:
        ValueError: Si FAISS_INDEX_TYPE n'est pas reconnu.
    """
    dim = embeddings.shape[1]

    if FAISS_INDEX_TYPE == "flat":
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    elif FAISS_INDEX_TYPE == "ivf":
        quantizer = faiss.IndexFlatL2(dim)
        index     = faiss.IndexIVFFlat(quantizer, dim, IVF_NLIST)
        print(f"  Entraînement de l'index IVF sur {len(embeddings)} vecteurs…")
        index.train(embeddings)
        index.add(embeddings)

    elif FAISS_INDEX_TYPE == "hnsw":
        index = faiss.IndexHNSWFlat(dim, HNSW_M)
        index.add(embeddings)

    else:
        raise ValueError(
            f"FAISS_INDEX_TYPE inconnu : '{FAISS_INDEX_TYPE}'. "
            "Valeurs acceptées : 'flat', 'ivf', 'hnsw'."
        )

    return index


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    vector_path.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des chunks
    texts, metadata = load_chunks()
    total           = len(texts)

    # 2. Reprise ou démarrage
    existing_embeddings, start_index = load_existing_embeddings()

    # 3. Chargement du modèle d'embeddings
    print(f"\nChargement du modèle : {EMBEDDING_MODEL}")
    print("  (premier lancement : téléchargement ~1,1 Go dans ~/.cache/huggingface/)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Modèle chargé. Dimension des vecteurs : {EMBEDDING_DIM}")

    # Vérification de cohérence entre le modèle et la config
    actual_dim = model.get_sentence_embedding_dimension()
    if actual_dim != EMBEDDING_DIM:
        print(
            f"  ⚠ ATTENTION : EMBEDDING_DIM configuré à {EMBEDDING_DIM} "
            f"mais le modèle produit des vecteurs de dimension {actual_dim}.\n"
            f"  Corrigez EMBEDDING_DIM dans 00_config_local.py → {actual_dim}"
        )

    # 4. Vectorisation
    print()
    embeddings_array = embed_texts_in_batches(
        model, clean_texts(texts), start_index, existing_embeddings
    )

    # 5. Sauvegarde finale des embeddings
    np.save(embeddings_file, embeddings_array)
    print(f"\nEmbeddings sauvegardés : {embeddings_file.resolve()}")
    print(f"  Forme de la matrice : {embeddings_array.shape}")

    # 6. Construction et sauvegarde de l'index FAISS
    print(f"Construction de l'index FAISS (type : {FAISS_INDEX_TYPE})…")
    index = build_faiss_index(embeddings_array)
    faiss.write_index(index, str(index_file))
    print(f"Index FAISS sauvegardé  : {index_file.resolve()}")
    print(f"  Vecteurs indexés : {index.ntotal}")

    # 7. Sauvegarde des métadonnées
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Métadonnées sauvegardées : {metadata_file.resolve()}")

    print("\n✓ Pipeline build_embeddings_local terminé avec succès.")
    print("  Étape suivante : python 04_rag_query_local.py")


if __name__ == "__main__":
    main()
