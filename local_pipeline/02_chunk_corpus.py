"""
02_chunk_corpus.py
==================
Étape 2 du pipeline RAG documentaire.

Rôle : Découpe le corpus texte en chunks de taille contrôlée (en tokens),
avec chevauchement entre chunks pour éviter de couper le contexte aux
frontières. Produit le fichier chunks.json consommé par 03_build_embeddings.py.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json          (ce script)
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → réponse synthétique via LLM
    ──
    tool_semantic_search.py  diagnostic de recherche sans LLM 
    (renvoie les références des documents présents dans le corpus les plus pertinents)
    tool_add_document.py     ajout incrémental d'un PDF au corpus

Structure de fichiers attendue :
    projet/
    ├── 02_chunk_corpus.py
    ├── extracted_text/
    │   ├── corpus.txt       ← produit par 01_extract_text.py (requis)
    │   └── chunks.json      ← produit par ce script
    └── vector_store/        ← produit par 03_build_embeddings.py

Format attendu de corpus.txt :
    Chaque section doit être introduite par un en-tête de la forme :
        [nom_du_fichier — page N]
        texte de la section...
    Exemple :
        [rapport_2024.pdf — page 3]
        Le chiffre d'affaires annuel s'élève à...
"""

# =============================================================================
# PARAMÈTRES — ajustez selon votre contexte
# =============================================================================

# --- Chemins ---
# Fichier texte brut produit par extract_corpus.py
INPUT_FILE = "extracted_text/corpus.txt"

# Fichier JSON de sortie consommé par build_embeddings.py
OUTPUT_FILE = "extracted_text/chunks.json"

# --- Tokenizer ---
# "cl100k_base" est le tokenizer des modèles OpenAI récents (GPT-4, embeddings v3).
# À ne changer que si vous utilisez un modèle avec un tokenizer différent.
TOKENIZER_ENCODING = "cl100k_base"

# --- Taille des chunks ---
# Nombre maximum de tokens par chunk.
# Valeurs recommandées selon le modèle d'embeddings :
#   text-embedding-3-small / large : fenêtre de 8191 tokens, mais 256–1024 tokens
#   est plus efficace en pratique pour la précision de la recherche.
#   512–800 tokens est un bon compromis entre contexte et précision.
MAX_TOKENS = 800

# Nombre de tokens de chevauchement entre deux chunks consécutifs.
# Le chevauchement évite de couper une phrase ou une idée à la frontière
# entre deux chunks. Valeur typique : 10–20 % de MAX_TOKENS.
# Exemple : avec MAX_TOKENS=800 et OVERLAP=150, un chunk commence
# 150 tokens avant la fin du chunk précédent.
OVERLAP = 150

# =============================================================================
# IMPORTS
# =============================================================================

import json
import re
import tiktoken
from pathlib import Path

# =============================================================================
# INITIALISATION
# =============================================================================

input_path  = Path(INPUT_FILE)
output_path = Path(OUTPUT_FILE)

encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)

# =============================================================================
# FONCTIONS
# =============================================================================

def count_tokens(text: str) -> int:
    """
    Compte le nombre de tokens d'un texte selon le tokenizer configuré.

    Args:
        text: Texte à mesurer.

    Returns:
        Nombre de tokens.
    """
    return len(encoding.encode(text))


def split_text(text: str) -> list[str]:
    """
    Découpe un texte en chunks de taille MAX_TOKENS avec chevauchement OVERLAP.

    Fonctionnement :
        - Le texte est d'abord encodé en tokens.
        - On extrait des fenêtres de MAX_TOKENS tokens.
        - Chaque fenêtre suivante commence OVERLAP tokens avant la fin
          de la fenêtre précédente, assurant la continuité du contexte.
        - Les tokens sont re-décodés en texte pour chaque chunk.

    Exemple avec MAX_TOKENS=800 et OVERLAP=150 :
        chunk 1 : tokens [0   → 799]
        chunk 2 : tokens [650 → 1449]
        chunk 3 : tokens [1300 → 2099]
        ...

    Args:
        text: Texte brut d'une section du corpus.

    Returns:
        Liste de chaînes de caractères, chacune représentant un chunk.
    """
    tokens = encoding.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + MAX_TOKENS
        chunk_text = encoding.decode(tokens[start:end])
        chunks.append(chunk_text)

        next_start = end - OVERLAP
        # Garde-fou : évite une boucle infinie si OVERLAP >= MAX_TOKENS
        start = max(next_start, start + 1)

    return chunks


def parse_corpus(raw_text: str) -> list[dict]:
    """
    Parse le corpus texte et découpe chaque section en chunks.

    Le corpus est structuré avec des en-têtes de la forme :
        [nom_source — page N]

    Chaque section (texte entre deux en-têtes) est découpée indépendamment
    via split_text(), puis chaque chunk est enrichi de ses métadonnées
    (source, page, nombre de tokens).

    Args:
        raw_text: Contenu brut du fichier corpus.txt.

    Returns:
        Liste de dicts avec les clés : source, page, text, tokens.
    """
    pattern = re.compile(r"\[(.+?) — page (\d+)\]\n", re.MULTILINE)
    matches = list(pattern.finditer(raw_text))

    if not matches:
        raise ValueError(
            "Aucun en-tête trouvé dans le corpus.\n"
            "Format attendu : [nom_source — page N]\\n"
        )

    chunks = []

    for i, match in enumerate(matches):
        source = match.group(1)
        page   = int(match.group(2))

        # Délimite le texte de la section entre cet en-tête et le suivant
        text_start = match.end()
        text_end   = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        text       = raw_text[text_start:text_end].strip()

        if not text:
            continue

        for chunk_text in split_text(text):
            chunks.append({
                "source": source,
                "page":   page,
                "text":   chunk_text,
                "tokens": count_tokens(chunk_text),
            })

    return chunks


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    if not input_path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {input_path.resolve()}\n"
            "Lancez d'abord extract_corpus.py pour générer ce fichier."
        )

    print(f"Lecture du corpus : {input_path.resolve()}")
    raw_text = input_path.read_text(encoding="utf-8")

    chunks = parse_corpus(raw_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    total_tokens = sum(c["tokens"] for c in chunks)
    print(f"{len(chunks)} chunks générés ({total_tokens:,} tokens au total)")
    print(f"Fichier de sortie : {output_path.resolve()}")
    print("\n✓ Pipeline chunk_corpus terminé avec succès.")


if __name__ == "__main__":
    main()
