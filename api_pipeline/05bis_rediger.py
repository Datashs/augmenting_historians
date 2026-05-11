"""
05bis_rediger.py
================
Étape 5bis du pipeline RAG — Rédaction assistée par consigne (VERSION OPENAI).

Rôle : À partir d'une consigne de rédaction (question historiographique,
angle d'analyse, plan partiel), recherche les passages pertinents dans
le corpus, puis demande au LLM de rédiger un passage académique structuré,
ancré dans les sources, avec citations explicites [source, p. N].

Pipeline :
    03_build_embeddings.py → vector_store/{…}
    05_confronter.py       → confrontation paragraphe ↔ corpus
    05bis_rediger.py       → rédaction assistée par consigne  (ce script)
    06_map_enrich.py       → cartographie enrichissement

Différence avec 05_confronter.py :
    - 05__rag_write : part d'un PARAGRAPHE existant → l'ancre dans le corpus
    - 05bis_rediger : part d'une CONSIGNE           → rédige un nouveau passage

Embeddings : OpenAI (text-embedding-3-large) — nécessite OPENAI_API_KEY
Génération : OpenAI ou Ollama (configurable dans config_00.py)

Usage :
    python 05bis_rediger.py
    # Entrez la consigne au prompt interactif.
    # Exemple : "Rédigez un paragraphe sur l'évolution de la notion
    #            d'infraction politique dans le droit de l'extradition
    #            au XIXe siècle, en montrant la rupture introduite par
    #            la Révolution française."
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

from config_00 import (
    LLMClient,
    TOP_K,
    VECTOR_DIR,
    CORPUS_FILE,
    EMBEDDING_MODEL,
    OPENAI_LLM_MODEL,
    LLM_TEMPERATURE,
    charger_corpus,
    recuperer_passage,
)

# TOP_K plus élevé pour la rédaction : contexte plus riche = passage plus nourri
TOP_K_REDIGER = 15

# Longueur minimale d'un passage pour être inclus (en caractères)
LONGUEUR_PASSAGE_MIN = 100

# Sauvegarder la réponse dans outputs/ ?
SAVE_OUTPUT = True
OUTPUT_DIR  = "outputs"

# Encodage de sortie
ENCODAGE_SORTIE = "utf-8"

# Tokens maximum pour la rédaction — indépendant de LLM_MAX_TOKENS de config_00.py.
# 3000 tokens ≈ 2000-2200 mots, largement suffisant pour un passage académique long.
# Ne pas dépasser 4096 avec gpt-4.1-mini.
MAX_TOKENS_REDIGER = 3000

# =============================================================================
# PROMPT SYSTÈME
# =============================================================================

SYSTEM_PROMPT = """Tu es un assistant historien expert en rédaction académique.
À partir d'une consigne et d'extraits du corpus, tu rédiges un passage
structuré, en prose continue, dans un registre académique.

Règles absolues :
- Rédige UNIQUEMENT en prose continue, sans listes ni tirets.
- Cite les sources entre crochets sous la forme [source, p. N] à chaque
  affirmation factuelle.
- Ne cite QUE des passages présents dans les extraits fournis.
- N'invente aucune information, aucun auteur, aucune date absents des extraits.
- Le passage doit être directement utilisable dans un manuscrit historique."""

USER_TEMPLATE = """CONSIGNE DE RÉDACTION :
{consigne}

EXTRAITS DU CORPUS (cite-les sous la forme [nom_fichier, p. N]) :

{extraits}

Rédige maintenant un passage académique en prose continue répondant à la consigne,
en t'appuyant sur les extraits fournis et en citant chaque source utilisée.

PASSAGE RÉDIGÉ :"""

# =============================================================================
# IMPORTS
# =============================================================================

import sys
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime

# =============================================================================
# INITIALISATION
# =============================================================================

vector_dir    = Path(VECTOR_DIR)
index_file    = vector_dir / "faiss.index"
metadata_file = vector_dir / "metadata.json"
output_dir    = Path(OUTPUT_DIR)

# =============================================================================
# FONCTIONS
# =============================================================================

def charger_index() -> tuple[faiss.Index, list[dict]]:
    """Charge l'index FAISS et les métadonnées."""
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path.resolve()}\n"
                "Lancez d'abord 03_build_embeddings.py."
            )
    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


def encoder_consigne(consigne: str, client_openai) -> np.ndarray:
    """
    Encode la consigne en vecteur via l'API OpenAI embeddings.

    Le modèle doit être identique à celui de 03_build_embeddings.py.

    Args:
        consigne      : Texte de la consigne de rédaction.
        client_openai : Client OpenAI initialisé.

    Returns:
        Vecteur float32 de forme (1, dim) prêt pour index.search().
    """
    response = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=consigne,
    )
    vecteur = np.array(response.data[0].embedding, dtype="float32")
    return vecteur.reshape(1, -1)


def rechercher_passages(
    index: faiss.Index,
    metadata: list[dict],
    vecteur: np.ndarray,
    corpus_text: str,
) -> list[dict]:
    """
    Recherche les TOP_K_REDIGER passages les plus pertinents.

    Returns:
        Liste de dicts {source, page, passage}, dédoublonnés.
    """
    _, indices = index.search(vecteur, TOP_K_REDIGER)

    passages = []
    vus      = set()

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue

        meta = metadata[idx]
        cle  = (meta["source"], meta["page"])
        if cle in vus:
            continue
        vus.add(cle)

        passage = recuperer_passage(corpus_text, meta["source"], meta["page"])
        if len(passage) < LONGUEUR_PASSAGE_MIN:
            continue

        passages.append({
            "source":  meta["source"],
            "page":    meta["page"],
            "passage": passage,
        })

    return passages


def formater_extraits(passages: list[dict], max_chars: int = 600) -> str:
    """
    Formate les passages pour le prompt, avec référence explicite.

    Chaque extrait est numéroté et présenté avec sa référence [source, p. N].
    Les passages sont tronqués à max_chars caractères pour maîtriser la taille
    du prompt et laisser suffisamment de tokens disponibles pour la rédaction.

    Args:
        passages  : Liste de dicts {source, page, passage}.
        max_chars : Longueur maximale de chaque passage (défaut : 600 caractères
                    ≈ 3-4 phrases, suffisant pour capter l'argument principal).

    Returns:
        Extraits formatés, prêts à insérer dans le prompt.
    """
    blocs = []
    for i, p in enumerate(passages, 1):
        ref    = f"[{p['source']}, p. {p['page']}]"
        texte  = p['passage']
        if len(texte) > max_chars:
            texte = texte[:max_chars].rsplit(' ', 1)[0] + "…"  # coupe proprement au mot
        blocs.append(f"Extrait {i} {ref} :\n{texte}")
    return "\n\n---\n\n".join(blocs)


def rediger(consigne: str, passages: list[dict], client_openai) -> str:
    """
    Soumet la consigne et les passages au LLM pour rédaction.

    Appel direct à l'API OpenAI (sans passer par LLMClient) pour utiliser
    MAX_TOKENS_REDIGER = 3000 au lieu de LLM_MAX_TOKENS de config_00.py (1024).
    Cela évite la troncature des passages longs sans modifier la config globale.

    Args:
        consigne      : Consigne de rédaction formulée par l'utilisateur.
        passages      : Passages récupérés par FAISS.
        client_openai : Client OpenAI initialisé.

    Returns:
        Passage rédigé par le LLM.
    """
    if not passages:
        return (
            "Aucun passage pertinent trouvé dans le corpus pour cette consigne.\n"
            "Suggestions :\n"
            "  - Reformulez la consigne avec des termes plus proches du corpus\n"
            "  - Augmentez TOP_K_REDIGER dans les paramètres"
        )

    extraits = formater_extraits(passages)
    user_msg = USER_TEMPLATE.format(consigne=consigne, extraits=extraits)

    response = client_openai.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS_REDIGER,   # 3000 — local à ce script
    )
    return response.choices[0].message.content


def sauvegarder(consigne: str, passage: str, passages: list[dict]) -> Path:
    """
    Sauvegarde la consigne, les sources consultées et le passage rédigé.

    Format :
        CONSIGNE
        --------
        [consigne]

        SOURCES CONSULTÉES
        ------------------
        [liste source + page]

        PASSAGE RÉDIGÉ
        --------------
        [texte]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath  = output_dir / f"redaction_openai_{timestamp}.txt"

    sources = "\n".join(
        f"  • {p['source']} (p. {p['page']})"
        for p in passages
    )

    content = (
        f"CONSIGNE\n{'─' * 60}\n{consigne}\n\n"
        f"SOURCES CONSULTÉES ({len(passages)} passages)\n{'─' * 60}\n"
        f"{sources}\n\n"
        f"PASSAGE RÉDIGÉ\n{'─' * 60}\n{passage}\n"
    )

    filepath.write_text(content, encoding=ENCODAGE_SORTIE)
    return filepath


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("  05bis — RÉDACTION ASSISTÉE PAR CONSIGNE (OpenAI)")
    print("=" * 60)
    print(f"  TOP_K      : {TOP_K_REDIGER}")
    print(f"  Embeddings : OpenAI ({EMBEDDING_MODEL})")
    print()

    # Chargement
    try:
        print("Chargement de l'index FAISS…")
        index, metadata = charger_index()
        print(f"  {index.ntotal} vecteurs dans l'index.")

        print("Chargement du corpus…")
        corpus_text = charger_corpus()

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

    # Client OpenAI (embeddings + génération)
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        client_openai = OpenAI()
        print(f"  Modèle LLM : {OPENAI_LLM_MODEL} | max_tokens : {MAX_TOKENS_REDIGER}")
    except Exception as e:
        print(f"\n❌ Erreur initialisation OpenAI : {e}")
        sys.exit(1)

    # Boucle interactive
    while True:
        print("\n" + "─" * 60)
        print("Entrez votre consigne de rédaction (ou 'q' pour quitter).")
        print("Exemple : 'Rédigez un paragraphe sur l'évolution de la notion")
        print("          d'infraction politique au XIXe siècle.'")
        consigne = input("\nConsigne : ").strip()

        if consigne.lower() in ("q", "quit", "exit"):
            print("Au revoir.")
            break

        if not consigne:
            print("Consigne vide, ignorée.")
            continue

        # Encodage + recherche
        print("\nEncodage de la consigne…")
        try:
            vecteur = encoder_consigne(consigne, client_openai)
        except Exception as e:
            print(f"❌ Erreur d'encodage : {e}")
            continue

        print(f"Recherche de {TOP_K_REDIGER} passages pertinents…")
        passages = rechercher_passages(index, metadata, vecteur, corpus_text)
        print(f"  {len(passages)} passages trouvés :")
        for p in passages:
            print(f"    • {p['source']} (p. {p['page']})")

        if not passages:
            print("Aucun passage pertinent. Reformulez la consigne.")
            continue

        # Rédaction
        print(f"\n--- Passage rédigé ---\n")
        try:
            texte = rediger(consigne, passages, client_openai)
            print(texte)
        except Exception as e:
            print(f"❌ Erreur LLM : {e}")
            continue

        # Sauvegarde
        if SAVE_OUTPUT:
            filepath = sauvegarder(consigne, texte, passages)
            print(f"\n💾 Sauvegardé : {filepath.resolve()}")


if __name__ == "__main__":
    main()
