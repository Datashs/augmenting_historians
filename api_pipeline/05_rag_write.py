"""
05_rag_write.py
===============
Étape 5 du pipeline RAG — Écriture assistée.

Rôle : Lit un paragraphe du manuscrit (fichier .txt), l'utilise comme
requête RAG pour trouver les passages du corpus les plus pertinents,
puis demande au LLM d'analyser chaque passage et de proposer comment
l'intégrer dans le texte.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → exploration thématique libre
    05_rag_write.py        → écriture assistée (paragraphe → citations)  (ce script)
    06_map_enrich.py       → cartographie enrichissement (manuscrit entier)
    07_map_critique.py     → cartographie critique (manuscrit entier)
    ──
    00_config.py           → configuration et client LLM partagés

Principe de fonctionnement :
    Le paragraphe du manuscrit N'EST PAS ajouté au corpus RAG.
    Il sert uniquement de REQUÊTE : il interroge le corpus sans en faire
    partie. Cette séparation est fondamentale — elle évite que le LLM
    confonde le texte en cours d'écriture avec les sources établies.

    Pour chaque passage trouvé par FAISS, le LLM produit :
      1. La citation pertinente extraite du passage
      2. Sa fonction argumentative par rapport au paragraphe
      3. Une proposition d'insertion rédigée dans le texte
      4. La référence complète pour renvoi au document source

╔══════════════════════════════════════════════════════════════╗
║  CHOIX DU BACKEND LLM                                       ║
╠══════════════════════════════════════════════════════════════╣
║  La génération de texte peut utiliser OpenAI ou Ollama.     ║
║  Pour changer de backend, ouvrez 00_config.py et modifiez : ║
║                                                              ║
║      LLM_BACKEND = "openai"   ← API OpenAI (payant)         ║
║      LLM_BACKEND = "ollama"   ← modèle local (gratuit)      ║
║                                                              ║
║  Attention : les embeddings utilisent toujours OpenAI.      ║
║  Le fichier .env avec OPENAI_API_KEY est donc toujours       ║
║  nécessaire, quel que soit le backend de génération choisi.  ║
╚══════════════════════════════════════════════════════════════╝

Ce que ce script ne fait PAS :
    - Il n'invente pas de citations (le prompt l'interdit explicitement)
    - Il ne modifie pas le corpus
    - Il ne traite pas un manuscrit entier (→ voir 06 et 07 pour cela)
    - Mais comme la réponse est générée par un prompt adressé à un LLM, 
    il est sage de vérifier, ce que permet le renvoi aux coordonnées des 
    fragments retenus.

Usage :
    python 05_rag_write.py --paragraphe mon_paragraphe.txt
    python 05_rag_write.py --paragraphe mon_paragraphe.txt --sortie analyse.txt
    python 05_rag_write.py --paragraphe mon_paragraphe.txt --top_k 5

Arguments :
    --paragraphe FICHIER   Fichier .txt contenant le paragraphe à analyser
                           (obligatoire)
    --sortie FICHIER       Fichier de sortie pour l'analyse (optionnel)
                           Si absent, l'analyse est affichée dans le terminal
    --top_k N              Nombre de passages récupérés (écrase TOP_K du config)

Structure de fichiers attendue :
    projet/
    ├── .env                        ← OPENAI_API_KEY=sk-... (si backend OpenAI)
    ├── 00_config.py
    ├── 05_rag_write.py             ← ce script
    ├── mon_paragraphe.txt          ← fichier d'entrée (créé par l'utilisateur)
    ├── extracted_text/
    │   └── corpus.txt
    └── vector_store/
        ├── faiss.index
        └── metadata.json

Format du fichier d'entrée (mon_paragraphe.txt) :
    Fichier texte brut encodé en UTF-8, contenant un seul paragraphe.
    Pas de mise en forme particulière requise.
    Exemple :
        La construction de la frontière franco-allemande au XIXe siècle
        ne peut être comprise indépendamment des dynamiques locales.
        Les populations frontalières développèrent des pratiques d'adaptation
        qui échappaient en grande partie au contrôle des États centraux.
"""

# =============================================================================
# PARAMÈTRES — ajustez selon votre contexte
# =============================================================================

# --- Langue des prompts système ---
# Langue dans laquelle le LLM reçoit ses instructions.
# "fr" → prompts en français
# "en" → prompts en anglais (légèrement plus performant avec certains modèles)
LANGUE_PROMPTS = "fr"

# --- Recherche ---
# Nombre de passages récupérés par FAISS (peut être surchargé via --top_k).
# Pour l'écriture assistée, 5–8 est un bon équilibre : assez de diversité
# sans noyer l'analyse dans des passages redondants.
# Ce paramètre prend la valeur de TOP_K dans 00_config.py si non surchargé.
TOP_K_LOCAL = None  # None = utilise TOP_K de 00_config.py

# --- Filtrage des passages ---
# Longueur minimale (en caractères) d'un passage pour être inclus dans l'analyse.
# Les passages trop courts (ex : légendes de figures, en-têtes) apportent peu.
# Réduire si des passages pertinents sont filtrés à tort.
LONGUEUR_PASSAGE_MIN = 100

# --- Sortie ---
# Encodage du fichier de sortie (si --sortie est utilisé)
ENCODAGE_SORTIE = "utf-8"

# =============================================================================
# PROMPTS SYSTÈME
# =============================================================================
# Les prompts définissent le rôle et les contraintes du LLM.
# Ils sont séparés du code pour faciliter les ajustements sans toucher
# à la logique du script.
#
# Deux versions sont disponibles : française et anglaise.
# La version active est sélectionnée via LANGUE_PROMPTS.

PROMPTS = {
    "fr": {
        "system": """Tu es un assistant spécialisé en histoire, expert en analyse documentaire.
Tu aides un historien à enrichir son manuscrit en identifiant des passages
du corpus scientifique qui peuvent renforcer son argumentation.

Règles absolues :
- Ne cite QUE des passages explicitement présents dans les extraits fournis.
- N'invente aucune citation, aucun auteur, aucune date.
- Si un extrait n'est pas pertinent pour le paragraphe, dis-le clairement.
- Reste ancré dans ce que les textes disent réellement.""",

        "user": """Voici un paragraphe d'un manuscrit historique :

--- PARAGRAPHE ---
{paragraphe}
--- FIN DU PARAGRAPHE ---

Voici des extraits du corpus scientifique récupérés par similarité sémantique :

{extraits}

Pour chaque extrait pertinent, fournis une analyse structurée en quatre points :

1. CITATION PERTINENTE
   Cite le passage exact (entre guillemets) qui peut être utile au paragraphe.

2. FONCTION ARGUMENTATIVE
   Explique précisément comment cette citation renforce, nuance ou complète
   l'argument du paragraphe (2–3 phrases).

3. PROPOSITION D'INSERTION
   Rédige une phrase ou deux montrant comment intégrer cette citation dans
   le paragraphe, avec une formule d'introduction appropriée au style historique.

4. RÉFÉRENCE DE RENVOI
   Indique : Fichier source : [nom du fichier] | Page : [numéro de page]

Si un extrait n'est pas pertinent pour ce paragraphe, indique simplement :
[Extrait non pertinent — raison brève]

Commence ton analyse."""
    },

    "en": {
        "system": """You are a history research assistant, expert in documentary analysis.
You help historians enrich their manuscripts by identifying passages
from the scientific corpus that can strengthen their argumentation.

Absolute rules:
- Only cite passages explicitly present in the provided excerpts.
- Do not invent any citation, author, or date.
- If an excerpt is not relevant to the paragraph, say so clearly.
- Stay grounded in what the texts actually say.""",

        "user": """Here is a paragraph from a historical manuscript:

--- PARAGRAPH ---
{paragraphe}
--- END OF PARAGRAPH ---

Here are excerpts from the scientific corpus retrieved by semantic similarity:

{extraits}

For each relevant excerpt, provide a structured analysis in four points:

1. RELEVANT CITATION
   Quote the exact passage (in quotation marks) that can be useful to the paragraph.

2. ARGUMENTATIVE FUNCTION
   Explain precisely how this citation reinforces, nuances or completes
   the paragraph's argument (2–3 sentences).

3. INSERTION PROPOSAL
   Write one or two sentences showing how to integrate this citation into
   the paragraph, with an appropriate introduction for historical writing style.

4. REFERENCE
   Indicate: Source file: [filename] | Page: [page number]

If an excerpt is not relevant to this paragraph, simply indicate:
[Excerpt not relevant — brief reason]

Begin your analysis."""
    }
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import sys
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime

# Import du module de configuration partagé
# Ce module fournit : LLMClient, TOP_K, VECTOR_DIR, CORPUS_FILE,
# charger_corpus(), recuperer_passage(), formater_extraits()
from config_00 import (
    LLMClient,
    TOP_K,
    VECTOR_DIR,
    EMBEDDING_MODEL,
    charger_corpus,
    recuperer_passage,
    formater_extraits,
)

# =============================================================================
# INITIALISATION
# =============================================================================

vector_dir    = Path(VECTOR_DIR)
index_file    = vector_dir / "faiss.index"
metadata_file = vector_dir / "metadata.json"

# =============================================================================
# FONCTIONS — CHARGEMENT
# =============================================================================

def charger_paragraphe(chemin: str) -> str:
    """
    Lit le fichier .txt contenant le paragraphe à analyser.

    Le fichier doit être encodé en UTF-8 et contenir un seul paragraphe.
    Les lignes vides en début et fin sont supprimées.

    Args:
        chemin : Chemin vers le fichier .txt du paragraphe.

    Returns:
        Contenu textuel du paragraphe, nettoyé.

    Raises:
        FileNotFoundError : Si le fichier est introuvable.
        ValueError        : Si le fichier est vide après nettoyage.
    """
    path = Path(chemin)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}\n"
            "Créez un fichier .txt contenant le paragraphe à analyser."
        )

    texte = path.read_text(encoding="utf-8").strip()

    if not texte:
        raise ValueError(
            f"Le fichier {chemin} est vide. "
            "Ajoutez le paragraphe à analyser."
        )

    return texte


def charger_index() -> tuple[faiss.Index, list[dict]]:
    """
    Charge l'index FAISS et les métadonnées depuis le vector store.

    Returns:
        index    : Index FAISS prêt pour la recherche.
        metadata : Liste de dicts {source, page, tokens}.

    Raises:
        FileNotFoundError : Si faiss.index ou metadata.json sont absents.
    """
    for path in (index_file, metadata_file):
        if not path.exists():
            raise FileNotFoundError(
                f"Fichier introuvable : {path.resolve()}\n"
                "Lancez d'abord 03_build_embeddings.py."
            )

    index    = faiss.read_index(str(index_file))
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    return index, metadata


# =============================================================================
# FONCTIONS — RECHERCHE
# =============================================================================

def encoder_paragraphe(paragraphe: str, client_openai) -> np.ndarray:
    """
    Encode le paragraphe en vecteur via l'API OpenAI embeddings.

    Le modèle utilisé doit être identique à celui de 03_build_embeddings.py.
    Un modèle différent produirait des vecteurs incompatibles avec l'index
    FAISS, ce qui rendrait les résultats de recherche sans sens.

    Note : l'embedding utilise toujours l'API OpenAI, même si le backend
    LLM est Ollama. Ollama ne propose pas (encore) d'API d'embedding
    standardisée compatible avec FAISS dans ce pipeline.

    Args:
        paragraphe   : Texte du paragraphe à encoder.
        client_openai: Client OpenAI initialisé (distinct du LLMClient).

    Returns:
        Vecteur float32 de forme (1, dim) prêt pour index.search().
    """
    response = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=paragraphe,
    )
    vecteur = np.array(response.data[0].embedding, dtype="float32")
    return vecteur.reshape(1, -1)


def rechercher_passages(
    index: faiss.Index,
    metadata: list[dict],
    vecteur: np.ndarray,
    corpus_text: str,
    top_k: int,
) -> list[dict]:
    """
    Recherche les passages les plus proches dans l'index FAISS.

    Pour chaque chunk trouvé, récupère le passage complet de la page source
    depuis corpus.txt (via recuperer_passage() de 00_config.py), ce qui
    donne davantage de contexte au LLM qu'un simple fragment.

    Les passages trop courts (< LONGUEUR_PASSAGE_MIN caractères) sont
    filtrés pour écarter les entrées peu informatives (légendes, en-têtes).

    Args:
        index       : Index FAISS chargé.
        metadata    : Métadonnées des vecteurs.
        vecteur     : Vecteur du paragraphe, forme (1, dim).
        corpus_text : Contenu complet de corpus.txt.
        top_k       : Nombre de chunks à récupérer.

    Returns:
        Liste de dicts {source, page, passage} triés par pertinence.
        Les doublons (même source + page) sont dédoublonnés.
    """
    _, indices = index.search(vecteur, top_k)

    passages   = []
    vus        = set()  # Pour éviter les doublons source+page

    for idx in indices[0]:
        if idx < 0 or idx >= len(metadata):
            continue  # Index invalide (peut arriver avec certains types FAISS)

        meta    = metadata[idx]
        cle     = (meta["source"], meta["page"])

        if cle in vus:
            continue  # Doublon : même page déjà récupérée
        vus.add(cle)

        passage = recuperer_passage(corpus_text, meta["source"], meta["page"])

        if len(passage) < LONGUEUR_PASSAGE_MIN:
            continue  # Passage trop court, peu informatif

        passages.append({
            "source":  meta["source"],
            "page":    meta["page"],
            "passage": passage,
        })

    return passages


# =============================================================================
# FONCTIONS — ANALYSE LLM
# =============================================================================

def analyser_paragraphe(
    paragraphe: str,
    passages: list[dict],
    llm: LLMClient,
) -> str:
    """
    Soumet le paragraphe et les passages au LLM pour analyse.

    Construit le prompt en fonction de la langue configurée (LANGUE_PROMPTS),
    puis appelle le LLM via LLMClient.generate() — compatible OpenAI et Ollama.

    Args:
        paragraphe : Texte du paragraphe du manuscrit.
        passages   : Liste de dicts {source, page, passage} trouvés par FAISS.
        llm        : Instance de LLMClient (00_config.py).

    Returns:
        Analyse textuelle produite par le LLM.
    """
    if not passages:
        return (
            "Aucun passage pertinent trouvé dans le corpus pour ce paragraphe.\n"
            "Suggestions :\n"
            "  - Vérifiez que le corpus est bien indexé (03_build_embeddings.py)\n"
            "  - Essayez d'augmenter TOP_K\n"
            "  - Le sujet du paragraphe est peut-être peu représenté dans le corpus"
        )

    # Sélection du prompt selon la langue configurée
    langue = LANGUE_PROMPTS if LANGUE_PROMPTS in PROMPTS else "fr"
    prompt = PROMPTS[langue]

    extraits_formates = formater_extraits(passages)

    user_message = prompt["user"].format(
        paragraphe=paragraphe,
        extraits=extraits_formates,
    )

    return llm.generate(prompt["system"], user_message)


# =============================================================================
# FONCTIONS — SORTIE
# =============================================================================

def formater_rapport(
    paragraphe: str,
    passages: list[dict],
    analyse: str,
    fichier_entree: str,
) -> str:
    """
    Formate le rapport final avec en-tête, paragraphe source et analyse.

    Le rapport est structuré pour être lisible directement dans le terminal
    ou sauvegardé dans un fichier texte.

    Args:
        paragraphe     : Paragraphe analysé.
        passages       : Passages récupérés (pour l'en-tête statistique).
        analyse        : Analyse produite par le LLM.
        fichier_entree : Nom du fichier source (pour l'en-tête).

    Returns:
        Rapport complet formaté en texte.
    """
    separateur = "═" * 60
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M")

    rapport = f"""{separateur}
  ANALYSE RAG — ÉCRITURE ASSISTÉE
  Fichier  : {fichier_entree}
  Date     : {horodatage}
  Passages : {len(passages)} extrait(s) analysé(s)
{separateur}

PARAGRAPHE SOUMIS :
{'-' * 40}
{paragraphe}
{'-' * 40}

ANALYSE :
{analyse}

{separateur}
  FIN DE L'ANALYSE
{separateur}
"""
    return rapport


def sauvegarder_rapport(rapport: str, chemin: str):
    """
    Sauvegarde le rapport dans un fichier texte.

    Args:
        rapport : Contenu textuel du rapport.
        chemin  : Chemin du fichier de sortie.
    """
    path = Path(chemin)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rapport, encoding=ENCODAGE_SORTIE)
    print(f"\n✓ Rapport sauvegardé : {path.resolve()}")


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Parsing des arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="05_rag_write.py — écriture assistée par RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 05_rag_write.py --paragraphe mon_paragraphe.txt
  python 05_rag_write.py --paragraphe mon_paragraphe.txt --sortie analyse.txt
  python 05_rag_write.py --paragraphe mon_paragraphe.txt --top_k 5
        """
    )
    parser.add_argument(
        "--paragraphe",
        required=True,
        metavar="FICHIER",
        help="Fichier .txt contenant le paragraphe à analyser (obligatoire)",
    )
    parser.add_argument(
        "--sortie",
        default=None,
        metavar="FICHIER",
        help="Fichier de sortie pour le rapport (optionnel, défaut : terminal)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        metavar="N",
        help=f"Nombre de passages à récupérer (défaut : TOP_K dans 00_config.py)",
    )
    args = parser.parse_args()

    # Résolution de TOP_K : argument CLI > TOP_K_LOCAL > TOP_K de 00_config.py
    top_k = args.top_k or TOP_K_LOCAL or TOP_K

    # -------------------------------------------------------------------------
    # En-tête
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  05 — ÉCRITURE ASSISTÉE PAR RAG")
    print("=" * 60)
    print(f"  Paragraphe : {args.paragraphe}")
    print(f"  TOP_K      : {top_k}")
    print(f"  Langue     : {LANGUE_PROMPTS}")
    print()

    # -------------------------------------------------------------------------
    # Chargement
    # -------------------------------------------------------------------------
    try:
        print("Chargement du paragraphe…")
        paragraphe = charger_paragraphe(args.paragraphe)
        print(f"  {len(paragraphe)} caractères chargés.")

        print("Chargement de l'index FAISS…")
        index, metadata = charger_index()
        print(f"  {index.ntotal} vecteurs dans l'index.")

        print("Chargement du corpus…")
        corpus_text = charger_corpus()

        print("Initialisation du client LLM…")
        llm = LLMClient()

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Embedding du paragraphe
    # Note : l'embedding utilise toujours OpenAI (voir encoder_paragraphe()).
    # -------------------------------------------------------------------------
    print("\nEncodage du paragraphe…")
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
        client_openai = OpenAI()
        vecteur = encoder_paragraphe(paragraphe, client_openai)
    except Exception as e:
        print(f"\n❌ Erreur d'encodage : {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Recherche FAISS
    # -------------------------------------------------------------------------
    print(f"Recherche des {top_k} passages les plus proches…")
    passages = rechercher_passages(index, metadata, vecteur, corpus_text, top_k)
    print(f"  {len(passages)} passage(s) exploitable(s) trouvé(s).")

    if passages:
        print("\n  Sources identifiées :")
        for p in passages:
            print(f"    • {p['source']} (p. {p['page']})")

    # -------------------------------------------------------------------------
    # Analyse LLM
    # -------------------------------------------------------------------------
    print("\nAnalyse en cours (LLM)…")
    try:
        analyse = analyser_paragraphe(paragraphe, passages, llm)
    except Exception as e:
        print(f"\n❌ Erreur LLM : {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Rapport
    # -------------------------------------------------------------------------
    rapport = formater_rapport(paragraphe, passages, analyse, args.paragraphe)

    if args.sortie:
        sauvegarder_rapport(rapport, args.sortie)
    else:
        print("\n" + rapport)

    print("\n✓ Analyse terminée.")


if __name__ == "__main__":
    main()
