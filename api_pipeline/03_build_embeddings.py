"""
03_build_embeddings.py
======================
Étape 3 du pipeline RAG documentaire.

Rôle : Génère les embeddings vectoriels de chaque chunk de texte via l'API
OpenAI, puis construit un index FAISS pour la recherche par similarité.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}  (ce script)
    04_rag_query.py        → réponse synthétique via LLM
    ──
    tool_semantic_search.py  Renvoie les référécens des segments les plus pertinents
    tool_add_document.py     ajout incrémental d'un PDF au corpus

Structure de fichiers attendue :
    projet/
    ├── .env                        ← OPENAI_API_KEY=sk-...
    ├── 03_build_embeddings.py
    ├── extracted_text/
    │   └── chunks.json             ← produit par 02_chunk_corpus.py (requis)
    └── vector_store/               ← créé automatiquement par ce script
        ├── embeddings.npy
        ├── faiss.index
        └── metadata.json

Prérequis :
    - pip install openai faiss-cpu numpy python-dotenv
    - Fichier .env à la racine du projet contenant : OPENAI_API_KEY=sk-...

Reprise automatique : si embeddings.npy existe déjà, le script reprend
depuis le dernier chunk traité sans tout recalculer.
Cela permet ne pas perdre de temps (et d'argent) en cas d'interuption de service
ou d'incident.
"""

# =============================================================================
# PARAMÈTRES — ajustez selon votre contexte
# =============================================================================

# --- Chemins ---
# Fichier produit par extract_chunks.py
CHUNKS_FILE = "extracted_text/chunks.json"

# Dossier de sortie (créé automatiquement s'il n'existe pas)
OUTPUT_DIR = "vector_store"

# --- Modèle d'embeddings OpenAI ---
# "text-embedding-3-small" : plus rapide, moins coûteux, dimension 1536
# "text-embedding-3-large" : meilleure qualité,  plus coûteux, dimension 3072
EMBEDDING_MODEL = "text-embedding-3-large"

# Dimension des vecteurs produits par le modèle choisi
# text-embedding-3-small → 1536  |  text-embedding-3-large → 3072
EMBEDDING_DIM = 3072

# --- Type d'index FAISS ---
# "flat"  : IndexFlatL2  — recherche exacte, recommandé jusqu'à ~50 000 chunks
# "ivf"   : IndexIVFFlat — rapide sur gros volumes (>50 000 chunks), moins précis
# "hnsw"  : IndexHNSWFlat — très rapide et précis, mais consomme plus de RAM
FAISS_INDEX_TYPE = "flat"

# Nombre de clusters pour IndexIVFFlat (ignoré pour "flat" et "hnsw")
# Règle empirique : nlist ≈ sqrt(nombre de chunks)
IVF_NLIST = 100

# Nombre de voisins pour IndexHNSWFlat (ignoré pour "flat" et "ivf")
# Plus M est élevé → meilleure précision, plus de RAM (valeurs typiques : 16–64)
HNSW_M = 32

# --- Gestion des erreurs et sauvegardes ---
# Pause (secondes) entre deux tentatives en cas de rate limit OpenAI
RATE_LIMIT_SLEEP = 1.0

# Pause (secondes) entre deux tentatives en cas d'erreur réseau
CONNECTION_ERROR_SLEEP = 5.0

# Sauvegarde intermédiaire tous les N chunks (protection contre les interruptions)
SAVE_EVERY_N_CHUNKS = 100

# Affichage de la progression tous les N chunks
LOG_EVERY_N_CHUNKS = 10

# =============================================================================
# IMPORTS
# =============================================================================

import json
import time
import numpy as np
import faiss
from pathlib import Path
from openai import OpenAI, RateLimitError, APIConnectionError, BadRequestError
from dotenv import load_dotenv

# =============================================================================
# INITIALISATION
# =============================================================================

load_dotenv()
client = OpenAI()  # Lit automatiquement OPENAI_API_KEY depuis .env

output_path   = Path(OUTPUT_DIR)
chunks_path   = Path(CHUNKS_FILE)
embeddings_file = output_path / "embeddings.npy"
index_file      = output_path / "faiss.index"
metadata_file   = output_path / "metadata.json"

# =============================================================================
# FONCTIONS
# =============================================================================

def clean_text(text: str) -> str:
    """
    Nettoie un texte avant de l'envoyer à l'API d'embeddings.

    - Remplace les textes vides par la chaîne sentinelle "vide"
      (l'API OpenAI refuse les chaînes vides).
    - Supprime les caractères nuls (\x00) qui provoquent des BadRequestError.
    - Filtre les caractères non imprimables (tout en conservant \n et \t).

    Args:
        text: Texte brut issu du chunk.

    Returns:
        Texte nettoyé, jamais vide.
    """
    if not text or not text.strip():
        return "vide"
    text = text.replace("\x00", "")
    text = "".join(c for c in text if c.isprintable() or c in "\n\t")
    return text.strip() or "vide"


def embed_text(text: str, chunk_index: int) -> list[float]:
    """
    Appelle l'API OpenAI pour obtenir le vecteur d'embedding d'un texte.

    Gère les erreurs transitoires par boucle de retry :
    - RateLimitError      → attend RATE_LIMIT_SLEEP secondes et réessaie
    - APIConnectionError  → attend CONNECTION_ERROR_SLEEP secondes et réessaie
    - BadRequestError     → chunk invalide, retourne un vecteur nul

    Args:
        text:        Texte à encoder (déjà nettoyé).
        chunk_index: Index du chunk (pour les messages de log).

    Returns:
        Vecteur d'embedding (liste de floats, longueur = EMBEDDING_DIM).
    """
    while True:
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
            return response.data[0].embedding

        except RateLimitError:
            print(f"  [rate limit] chunk {chunk_index + 1} — attente {RATE_LIMIT_SLEEP}s…")
            time.sleep(RATE_LIMIT_SLEEP)

        except APIConnectionError:
            print(f"  [réseau] chunk {chunk_index + 1} — nouvelle tentative dans {CONNECTION_ERROR_SLEEP}s…")
            time.sleep(CONNECTION_ERROR_SLEEP)

        except BadRequestError as e:
            print(f"  [invalide] chunk {chunk_index + 1} ignoré : {e}")
            return [0.0] * EMBEDDING_DIM


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Construit un index FAISS à partir d'une matrice d'embeddings.

    Le type d'index est contrôlé par le paramètre FAISS_INDEX_TYPE :
    - "flat"  : IndexFlatL2  — exact, recommandé jusqu'à ~50 000 chunks
    - "ivf"   : IndexIVFFlat — clustering, rapide sur grands corpus
    - "hnsw"  : IndexHNSWFlat — graphe de proximité, rapide et précis

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
        index = faiss.IndexIVFFlat(quantizer, dim, IVF_NLIST)
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


def load_chunks() -> tuple[list[str], list[dict]]:
    """
    Charge le fichier chunks.json produit par extract_chunks.py.

    Format attendu : liste de dicts avec les clés
        - "text"   : contenu textuel du chunk
        - "source" : nom du fichier source
        - "page"   : numéro de page
        - "tokens" : nombre de tokens estimé

    Returns:
        texts    : liste des textes (un par chunk)
        metadata : liste des métadonnées associées
    """
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {chunks_path.resolve()}\n"
            "Lancez d'abord extract_chunks.py pour générer ce fichier."
        )

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    texts    = [c["text"] for c in chunks]
    metadata = [{"source": c["source"], "page": c["page"], "tokens": c["tokens"]}
                for c in chunks]

    print(f"{len(texts)} chunks chargés depuis {chunks_path}")
    return texts, metadata


def load_existing_embeddings() -> tuple[list, int]:
    """
    Charge les embeddings déjà calculés pour reprendre un traitement interrompu.

    Returns:
        embeddings  : liste des vecteurs déjà calculés (vide si aucun)
        start_index : index du prochain chunk à traiter
    """
    if embeddings_file.exists():
        embeddings = list(np.load(embeddings_file))
        start_index = len(embeddings)
        print(f"Reprise détectée : {start_index} embeddings existants.")
        return embeddings, start_index

    return [], 0


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Chargement des chunks
    texts, metadata = load_chunks()
    total = len(texts)

    # 2. Reprise ou démarrage
    embeddings, start_index = load_existing_embeddings()

    if start_index == 0:
        print(f"Génération des embeddings pour {total} chunks (modèle : {EMBEDDING_MODEL})…")
    else:
        print(f"Reprise depuis le chunk {start_index + 1} / {total}…")

    # 3. Calcul des embeddings chunk par chunk
    for i, text in enumerate(texts[start_index:], start=start_index):
        vector = embed_text(clean_text(text), i)
        embeddings.append(vector)

        if (i + 1) % LOG_EVERY_N_CHUNKS == 0 or (i + 1) == total:
            print(f"  {i + 1} / {total}")

        if (i + 1) % SAVE_EVERY_N_CHUNKS == 0:
            np.save(embeddings_file, np.array(embeddings, dtype="float32"))
            print(f"  Sauvegarde intermédiaire ({i + 1} chunks)")

    # 4. Sauvegarde finale des embeddings
    embeddings_array = np.array(embeddings, dtype="float32")
    np.save(embeddings_file, embeddings_array)
    print(f"\nEmbeddings sauvegardés : {embeddings_file.resolve()}")

    # 5. Construction et sauvegarde de l'index FAISS
    print(f"Construction de l'index FAISS (type : {FAISS_INDEX_TYPE})…")
    index = build_faiss_index(embeddings_array)
    faiss.write_index(index, str(index_file))
    print(f"Index FAISS sauvegardé  : {index_file.resolve()}")

    # 6. Sauvegarde des métadonnées
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Métadonnées sauvegardées : {metadata_file.resolve()}")

    print("\n✓ Pipeline build_embeddings terminé avec succès.")


if __name__ == "__main__":
    main()
