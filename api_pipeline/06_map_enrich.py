"""
06_map_enrich.py
================
Étape 6 du pipeline RAG — Cartographie d'enrichissement.

Rôle : Lit un manuscrit complet (.txt), le découpe en paragraphes,
interroge le corpus RAG pour chacun, demande au LLM d'évaluer la
pertinence des passages trouvés, et produit un fichier JSON structuré
consommé ensuite par 08_visualise.py pour la visualisation HTML.

Pipeline :
    01_extract_text.py     → extracted_text/corpus.txt
    02_chunk_corpus.py     → extracted_text/chunks.json
    03_build_embeddings.py → vector_store/{embeddings.npy, faiss.index, metadata.json}
    04_rag_query.py        → exploration thématique libre
    05_rag_write.py        → écriture assistée (paragraphe isolé)
    06_map_enrich.py       → cartographie enrichissement             (ce script)
    07_map_critique.py     → cartographie critique
    08_visualise.py        → visualisation HTML interactive
    ──
    00_config.py           → configuration et client LLM partagés

╔══════════════════════════════════════════════════════════════╗
║  CHOIX DU BACKEND LLM                                       ║
╠══════════════════════════════════════════════════════════════╣
║  Ce script boucle sur TOUS les paragraphes du manuscrit.    ║
║  Ollama (local) est recommandé pour éviter les coûts API.   ║
║  Pour changer de backend, ouvrez 00_config.py et modifiez : ║
║                                                              ║
║      LLM_BACKEND = "openai"   ← API OpenAI (payant)         ║
║      LLM_BACKEND = "ollama"   ← modèle local (gratuit)      ║
║                                                              ║
║  Attention : les embeddings utilisent toujours OpenAI.      ║
║  Le fichier .env avec OPENAI_API_KEY est donc toujours       ║
║  nécessaire, quel que soit le backend de génération choisi.  ║
║                                                              ║
║  Avec Ollama : définissez --pause 0.0 (pas de rate limit).  ║
╚══════════════════════════════════════════════════════════════╝

Principe fondamental (rappel) :
    Le manuscrit N'EST PAS intégré au corpus RAG. Il sert uniquement de
    source de REQUÊTES. Chaque paragraphe interroge le corpus comme une
    question — le corpus répond avec ses passages les plus proches.
    Cette séparation garantit que le LLM ne confond jamais le texte en
    cours d'écriture avec les sources établies.

Mode "enrichir" (ce script) :
    Pour chaque paragraphe, le LLM cherche CE QUI MANQUE et CE QUI POURRAIT
    RENFORCER l'argument. Il évalue positivement : quelles références du
    corpus permettraient de densifier le propos, d'appuyer une affirmation,
    d'ajouter une nuance documentée.
    → Posture : "Qu'est-ce que le corpus peut APPORTER à ce paragraphe ?"

Mode "critiquer" (07_map_critique.py) :
    Même mécanique, mais posture inversée : le LLM cherche les failles,
    les affirmations non étayées, les contradictions avec le corpus.
    → Posture : "Qu'est-ce que le corpus REMET EN QUESTION dans ce paragraphe ?"

LE SCORE DE DENSITÉ DOCUMENTAIRE — comment le lire et comment il est calculé
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    Ce score est la métrique centrale du script. Il mesure dans quelle mesure
    le corpus scientifique disponible peut ENRICHIR un paragraphe donné du
    manuscrit. Il est exprimé entre 0.0 (aucune ressource utile) et 1.0
    (corpus très riche sur le sujet exact du paragraphe).

    Comment il est produit — les trois étapes :

    1. RECHERCHE SÉMANTIQUE (FAISS)
       Le paragraphe est converti en vecteur numérique (embedding).
       FAISS compare ce vecteur à l'ensemble du corpus et retourne les
       TOP_K passages les plus proches sémantiquement. Cette proximité
       est mathématique — elle mesure la similarité de sens, pas de mots.
       Un passage peut être pertinent sans partager un seul mot avec le
       paragraphe (ex : "frontière" et "limite territoriale").

    2. ÉVALUATION PAR LE LLM
       Le LLM reçoit le paragraphe ET les passages trouvés. Il évalue
       la pertinence réelle de chaque passage pour le propos spécifique
       du paragraphe. C'est ici que la machine dépasse la simple similarité
       vectorielle : le LLM comprend le contexte argumentatif.
       Il attribue un score de 0 à 10 selon la richesse documentaire
       qu'il perçoit dans les passages fournis.

    3. NORMALISATION
       Le score brut (0–10) est divisé par 10 pour obtenir un flottant
       entre 0.0 et 1.0, homogène avec les scores du script 07.

    Ce que le score mesure précisément :
       Score élevé (≥ 0.7) → le corpus contient des ressources pertinentes
         pour enrichir ce paragraphe. Des citations, nuances ou références
         supplémentaires sont disponibles et exploitables.
       Score moyen (0.4–0.7) → quelques ressources utiles existent mais
         le corpus est partiellement couvert sur ce sujet.
       Score faible (< 0.4) → le corpus apporte peu à ce paragraphe.
         Deux interprétations possibles : soit le paragraphe est déjà bien
         étayé et autonome, soit le corpus ne couvre pas ce sujet — le
         contexte et l'analyse LLM permettent de distinguer les deux cas.

    Ce que le score NE mesure PAS :
       ✗ La qualité intrinsèque du paragraphe (un paragraphe excellent
         peut avoir un score faible si le corpus ne le couvre pas).
       ✗ La vérité des affirmations (→ voir script 07 pour la critique).
       ✗ La complétude de la littérature existante sur le sujet
         (il mesure uniquement ce qui est dans VOTRE corpus indexé).

    Limites et précautions d'usage :
       Le score repose sur le jugement du LLM, qui peut varier d'un appel
       à l'autre (effet de la température) et dépend de la qualité des
       passages récupérés par FAISS. Il doit être traité comme un SIGNAL
       d'orientation, non comme une mesure absolue. Un score de 0.3 ne
       signifie pas que le paragraphe est mauvais — il signifie que le
       corpus indexé n'offre pas beaucoup de ressources supplémentaires
       pour ce passage précis.

Sortie produite :
    Un fichier JSON (SORTIE_JSON) contenant pour chaque paragraphe :
    - son index et son texte
    - son score de densité documentaire (0.0 à 1.0)
    - les passages trouvés avec leurs métadonnées
    - l'analyse LLM complète
    Ce JSON est la source unique de vérité pour 08_visualise.py.

Reprises automatiques :
    Si le fichier JSON de sortie existe déjà et contient des résultats
    partiels, le script reprend depuis le dernier paragraphe traité.
    Utile si le traitement est interrompu (réseau, erreur LLM, etc.).

Usage :
    python 06_map_enrich.py --manuscrit mon_manuscrit.txt
    python 06_map_enrich.py --manuscrit mon_manuscrit.txt --sortie resultats/enrich.json
    python 06_map_enrich.py --manuscrit mon_manuscrit.txt --top_k 6 --pause 1.5

Arguments :
    --manuscrit FICHIER   Fichier .txt du manuscrit complet (obligatoire)
    --sortie FICHIER      Fichier JSON de sortie (défaut : voir SORTIE_JSON)
    --top_k N             Nombre de passages FAISS par paragraphe
    --pause N             Pause en secondes entre deux appels LLM (défaut : 1.0)
                          Utile pour éviter les rate limits OpenAI
    --min_chars N         Longueur minimale d'un paragraphe à traiter

Structure de fichiers attendue :
    projet/
    ├── .env
    ├── 00_config.py
    ├── 06_map_enrich.py        ← ce script
    ├── mon_manuscrit.txt       ← fichier d'entrée
    ├── resultats/              ← créé automatiquement
    │   └── enrich.json         ← sortie consommée par 08_visualise.py
    ├── extracted_text/
    │   └── corpus.txt
    └── vector_store/
        ├── faiss.index
        └── metadata.json
"""

# =============================================================================
# PARAMÈTRES — modifiez uniquement cette section
# =============================================================================

# --- Chemins ---
# Fichier JSON de sortie (consommé par 08_visualise.py)
# Peut être surchargé via --sortie
SORTIE_JSON = "resultats/enrich.json"

# --- Segmentation du manuscrit ---
# Longueur minimale d'un paragraphe pour être analysé (en caractères).
# Les paragraphes trop courts (titres, intertitres, lignes isolées) sont
# ignorés : ils ne contiennent pas assez de substance argumentative pour
# une recherche sémantique pertinente.
# Réduire si des paragraphes courts mais importants sont ignorés à tort.
PARAGRAPHE_MIN_CHARS = 150

# Nombre de lignes vides consécutives pour délimiter un paragraphe.
# 1 = chaque saut de ligne simple crée un nouveau paragraphe
# 2 = deux sauts de ligne (ligne vide) créent un nouveau paragraphe (standard)
# Ajustez selon la convention typographique de votre manuscrit.
SAUTS_LIGNE_SEPARATEUR = 2

# --- Recherche ---
# Nombre de passages FAISS récupérés par paragraphe.
# Peut être surchargé via --top_k.
# Pour l'enrichissement : 6–8 est un bon équilibre.
# Plus TOP_K est élevé → analyse plus riche mais prompts plus longs.
TOP_K_LOCAL = None  # None = utilise TOP_K de 00_config.py

# --- Filtrage des passages ---
# Longueur minimale d'un passage pour être soumis au LLM (en caractères).
LONGUEUR_PASSAGE_MIN = 100

# --- Cadence ---
# Pause en secondes entre deux appels LLM.
# Recommandé avec OpenAI pour éviter les rate limits (0.5–2.0 secondes).
# Avec Ollama en local : mettre 0.0 (pas de limite réseau).
PAUSE_ENTRE_APPELS = 1.0

# --- Langue des prompts ---
# "fr" → instructions en français
# "en" → instructions en anglais
LANGUE_PROMPTS = "fr"

# --- Sauvegarde ---
# Sauvegarde intermédiaire du JSON tous les N paragraphes.
# Protège contre les interruptions sur les longs manuscrits.
SAUVEGARDER_TOUS_LES_N = 5

# =============================================================================
# PROMPTS SYSTÈME
# =============================================================================
# Le prompt d'enrichissement invite le LLM à adopter une posture
# constructive : trouver ce que le corpus peut APPORTER au paragraphe.
# Il est distinct du prompt de critique (07) qui adopte une posture
# adversariale.
#
# Structure de la réponse demandée :
#   SCORE      : densité documentaire perçue (0–10), traduit en 0.0–1.0
#   PASSAGES   : évaluation de chaque passage trouvé
#   MANQUES    : ce qui serait utile mais absent du corpus
#   SYNTHESE   : résumé actionnable pour l'historien

PROMPTS = {
    "fr": {
        "system": """Tu es un assistant de recherche en histoire, spécialisé dans
l'analyse documentaire et l'enrichissement de manuscrits scientifiques.

Ta mission : évaluer dans quelle mesure les extraits du corpus peuvent
enrichir et renforcer le paragraphe soumis.

Règles absolues :
- Évalue UNIQUEMENT sur la base des extraits fournis.
- Ne cite que des passages explicitement présents dans les extraits.
- N'invente aucune référence, aucun auteur, aucune date.
- Sois précis, concis, actionnable.
- Attribue un score honnête : un paragraphe peut très bien ne pas
  trouver de renforcement utile dans le corpus fourni.""",

        "user": """Voici un paragraphe d'un manuscrit historique :

--- PARAGRAPHE (index {index}) ---
{paragraphe}
--- FIN ---

Voici les extraits du corpus récupérés par similarité sémantique :

{extraits}

TÂCHE EN DEUX PARTIES :

━━━ PARTIE 1 — PASSAGES UTILES (JSON) ━━━
Pour chaque extrait véritablement utile à ce paragraphe, produis un objet JSON.
Réponds d'abord avec un bloc JSON valide, sans texte avant ni après, sans backticks :

[
  {{
    "source": "fichier.pdf",
    "page": N,
    "utile": true,
    "apport": "Phrase 1 : citation directe de l'extrait entre guillemets simples ('...'). Phrase 2 : ce que cet apport concret peut apporter au paragraphe.",
    "usage": "citation | note | référence"
  }},
  ...
]

Règles pour l'apport :
- Exactement 2 phrases.
- Phrase 1 : intègre obligatoirement une citation directe de l'extrait entre guillemets simples (1 à 2 phrases complètes, non tronquées).
- Phrase 2 : explique précisément ce que cet extrait peut apporter au paragraphe.
- Si l'extrait n'est pas utile : utile = false, apport = "Extrait non pertinent pour ce paragraphe.", usage = "".

━━━ PARTIE 2 — SCORE, MANQUES ET SYNTHÈSE (texte) ━━━
Après le JSON, sur une nouvelle ligne :

SCORE: X/10
(0 = aucun extrait pertinent | 10 = corpus très riche sur ce sujet précis)

MANQUES: [2-3 pistes de recherche formulées en une phrase chacune, séparées par des tirets]

SYNTHESE: En 2-3 phrases : bilan actionnable. Que faire en priorité pour renforcer ce paragraphe ?"""
    },

    "en": {
        "system": """You are a history research assistant, specialized in documentary
analysis and scientific manuscript enrichment.

Your mission: evaluate how much the corpus excerpts can enrich and
strengthen the submitted paragraph.

Absolute rules:
- Evaluate ONLY based on the provided excerpts.
- Only cite passages explicitly present in the excerpts.
- Do not invent any reference, author, or date.
- Be precise, concise, actionable.
- Give an honest score: a paragraph may simply not find useful
  reinforcement in the provided corpus.""",

        "user": """Here is a paragraph from a historical manuscript:

--- PARAGRAPH (index {index}) ---
{paragraphe}
--- END ---

Here are corpus excerpts retrieved by semantic similarity:

{extraits}

Produce a structured analysis in four sections:

DENSITY SCORE [0-10]
Evaluate the documentary richness available in the corpus for this paragraph.
0 = no relevant excerpt | 10 = corpus very rich on this precise subject
Format: SCORE: X/10

USEFUL PASSAGES
For each excerpt genuinely useful to this paragraph:
  - SOURCE: [file] (p. [page])
  - CONTRIBUTION: what this excerpt concretely brings (1–2 sentences)
  - SUGGESTED USE: how to integrate it (citation, footnote, reference)
If no excerpt is useful, write: "No directly usable excerpt."

IDENTIFIED GAPS
What types of sources or arguments would be useful but absent from the corpus?
(2–3 points maximum, formulated as research leads)

SYNTHESIS
In 2–3 sentences: actionable summary for the historian.
What should be done first to strengthen this paragraph?"""
    }
}

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import re
import sys
import time
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

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

load_dotenv()

vector_dir    = Path(VECTOR_DIR)
index_file    = vector_dir / "faiss.index"
metadata_file = vector_dir / "metadata.json"

# =============================================================================
# FONCTIONS — SEGMENTATION DU MANUSCRIT
# =============================================================================

def charger_manuscrit(chemin: str) -> str:
    """
    Lit le fichier .txt du manuscrit complet.

    Args:
        chemin : Chemin vers le fichier .txt du manuscrit.

    Returns:
        Contenu textuel brut du manuscrit.

    Raises:
        FileNotFoundError : Si le fichier est introuvable.
        ValueError        : Si le fichier est vide.
    """
    path = Path(chemin)
    if not path.exists():
        raise FileNotFoundError(
            f"Manuscrit introuvable : {path.resolve()}\n"
            "Vérifiez le chemin passé via --manuscrit."
        )

    texte = path.read_text(encoding="utf-8").strip()

    if not texte:
        raise ValueError(f"Le fichier {chemin} est vide.")

    return texte


def segmenter_manuscrit(texte: str, min_chars: int) -> list[dict]:
    """
    Découpe le manuscrit en paragraphes exploitables.

    Stratégie de découpage :
        Les paragraphes sont délimités par SAUTS_LIGNE_SEPARATEUR lignes
        vides consécutives (configurable). C'est la convention standard
        pour les fichiers texte issus de traitements de texte exportés.

        Les paragraphes trop courts (< min_chars caractères) sont ignorés :
        ils correspondent typiquement à des titres de sections, des
        intertitres, des notes isolées, ou des artefacts de mise en page.
        Ces éléments n'ont pas assez de substance pour une recherche
        sémantique pertinente.

    Args:
        texte     : Contenu brut du manuscrit.
        min_chars : Longueur minimale en caractères pour retenir un paragraphe.

    Returns:
        Liste de dicts {index, texte, nb_chars} dans l'ordre du manuscrit.
        L'index est continu et commence à 1 (plus lisible pour l'historien).
    """
    # Construction du séparateur regex selon SAUTS_LIGNE_SEPARATEUR
    # Ex : 2 sauts → \n\s*\n (une ligne vide entre deux blocs)
    separateur = r"\n" + r"\s*\n" * (SAUTS_LIGNE_SEPARATEUR - 1)
    blocs = re.split(separateur, texte)

    paragraphes = []
    index = 1

    for bloc in blocs:
        bloc_nettoye = bloc.strip()

        # Ignorer les blocs trop courts
        if len(bloc_nettoye) < min_chars:
            continue

        paragraphes.append({
            "index":    index,
            "texte":    bloc_nettoye,
            "nb_chars": len(bloc_nettoye),
        })
        index += 1

    return paragraphes


# =============================================================================
# FONCTIONS — RECHERCHE FAISS
# =============================================================================

def encoder_texte(texte: str, client_openai: OpenAI) -> np.ndarray:
    """
    Encode un texte en vecteur via l'API OpenAI embeddings.

    Identique à la fonction du script 05 : l'embedding utilise toujours
    OpenAI, même si le backend LLM est Ollama. L'index FAISS a été
    construit avec les vecteurs OpenAI — la requête doit utiliser
    le même espace vectoriel.

    Args:
        texte         : Texte à encoder.
        client_openai : Client OpenAI dédié aux embeddings.

    Returns:
        Vecteur float32 de forme (1, dim).
    """
    response = client_openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texte,
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
    Recherche les passages les plus proches dans FAISS pour un paragraphe.

    Logique identique au script 05, centralisée ici pour éviter la
    duplication. Les doublons (même source + page) sont dédoublonnés
    pour ne pas soumettre deux fois le même passage au LLM.

    Args:
        index       : Index FAISS chargé.
        metadata    : Métadonnées associées aux vecteurs.
        vecteur     : Vecteur du paragraphe, forme (1, dim).
        corpus_text : Contenu complet de corpus.txt.
        top_k       : Nombre de chunks à récupérer.

    Returns:
        Liste de dicts {source, page, passage}, dédoublonnée, filtrée.
    """
    distances, indices = index.search(vecteur, top_k)

    passages = []
    vus      = set()

    for rang, idx in enumerate(indices[0]):
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
            "source":   meta["source"],
            "page":     meta["page"],
            "passage":  passage,
            # La distance FAISS est conservée dans le JSON pour usage
            # éventuel dans la visualisation (tri, filtrage)
            "distance": float(distances[0][rang]),
        })

    return passages


# =============================================================================
# FONCTIONS — ANALYSE LLM
# =============================================================================

def extraire_score(analyse: str) -> float:
    """
    Extrait le score numérique depuis la réponse textuelle du LLM.

    Le LLM est invité à produire une ligne "SCORE: X/10".
    Cette fonction parse ce format et retourne un float entre 0.0 et 1.0.

    Stratégie de robustesse :
        Si le score n'est pas trouvé (format inattendu, LLM récalcitrant),
        la fonction retourne 0.5 comme valeur neutre plutôt que de planter.
        Un warning est affiché pour alerter l'utilisateur.

    Args:
        analyse : Texte brut de la réponse LLM.

    Returns:
        Score normalisé entre 0.0 et 1.0.
    """
    # # Pattern élargi — capture toutes les variantes produites par le LLM :
    # "SCORE: 8/10", "SCORE DE DENSITÉ: 8/10", "SCORE DE DENSITÉ : 7/10"
    match = re.search(r"SCORE(?:[^:\n]*)?\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10", analyse, re.IGNORECASE)

    if match:
        score_brut = float(match.group(1))
        # Clamp entre 0 et 10 avant normalisation (LLM parfois hors bornes)
        score_brut = max(0.0, min(10.0, score_brut))
        return round(score_brut / 10.0, 2)

    # Fallback : score non trouvé
    print("    ⚠️  Score non extrait de la réponse LLM — valeur neutre 0.5 utilisée.")
    return 0.5


def analyser_paragraphe(
    para: dict,
    passages: list[dict],
    llm: LLMClient,
    top_k_effectif: int,
) -> dict:
    """
    Soumet un paragraphe et ses passages au LLM pour analyse d'enrichissement.

    Construit le prompt selon LANGUE_PROMPTS, appelle le LLM, extrait le
    score et retourne un dict structuré prêt pour le JSON de sortie.

    Args:
        para           : Dict {index, texte, nb_chars} du paragraphe.
        passages       : Passages trouvés par FAISS pour ce paragraphe.
        llm            : Instance de LLMClient.
        top_k_effectif : Valeur de TOP_K utilisée (pour traçabilité).

    Returns:
        Dict structuré contenant toutes les informations du paragraphe
        analysé, prêt à être ajouté à la liste des résultats JSON.
    """
    # Cas sans passages : pas d'appel LLM, score minimal
    if not passages:
        return {
            "index":          para["index"],
            "texte":          para["texte"],
            "nb_chars":       para["nb_chars"],
            "score":          0.0,
            "passages":       [],
            "analyse":        "Aucun passage trouvé dans le corpus pour ce paragraphe.",
            "top_k":          top_k_effectif,
            "horodatage":     datetime.now().isoformat(),
        }

    # Construction du prompt
    langue = LANGUE_PROMPTS if LANGUE_PROMPTS in PROMPTS else "fr"
    prompt = PROMPTS[langue]

    extraits_formates = formater_extraits(passages)

    user_message = prompt["user"].format(
        index=para["index"],
        paragraphe=para["texte"],
        extraits=extraits_formates,
    )

    # Appel LLM
    analyse = llm.generate(prompt["system"], user_message)

    # Extraction du score
    score = extraire_score(analyse)

    # ── Parsing du JSON des passages structurés ────────────────────────────────
    passages_structures = []
    try:
        debut = analyse.find("[")
        fin   = analyse.rfind("]")
        if debut != -1 and fin != -1 and fin > debut:
            json_brut = analyse[debut:fin+1]
            items = json.loads(json_brut)
            for item in items:
                if item.get("utile", False):
                    passages_structures.append({
                        "source": item.get("source", "?"),
                        "page":   item.get("page", 0),
                        "apport": item.get("apport", ""),
                        "usage":  item.get("usage", ""),
                    })
    except (json.JSONDecodeError, ValueError) as e:
        print(f"    ⚠️  Parsing JSON passages échoué ({e}) — passages non structurés.")

    # Extraction des manques
    manques = []
    match_manques = re.search(r"MANQUES\s*:\s*(.+?)(?:SYNTHESE|SYNTHÈSE|\Z)", analyse, re.DOTALL | re.IGNORECASE)
    if match_manques:
        bloc_manques = match_manques.group(1).strip()
        manques = [m.strip("- •\t").strip() for m in bloc_manques.split("\n") if m.strip("- •\t").strip()]

    # Extraction de la synthèse
    synthese = ""
    match_synthese = re.search(r"(?:SYNTHESE|SYNTHÈSE)\s*:\s*(.+?)(?:\n\n|\Z)", analyse, re.DOTALL | re.IGNORECASE)
    if match_synthese:
        synthese = match_synthese.group(1).strip()

    return {
        "index":               para["index"],
        "texte":               para["texte"],
        "nb_chars":            para["nb_chars"],
        "score":               score,
        # Passages bruts pour le 08 et le 09 (compatibilité)
        "passages":            [{"source": p["source"], "page": p["page"],
                                 "distance": p["distance"]} for p in passages],
        # NOUVEAU — passages structurés avec apport cité + usage
        "passages_enrichis":   passages_structures,
        # NOUVEAU — manques et synthèse
        "manques":             manques,
        "synthese":            synthese,
        # Analyse brute conservée pour le 08
        "analyse":             analyse,
        "top_k":               top_k_effectif,
        "horodatage":          datetime.now().isoformat(),
    }


# =============================================================================
# FONCTIONS — GESTION DU JSON DE SORTIE (reprise)
# =============================================================================

def charger_resultats_existants(chemin: str) -> list[dict]:
    """
    Charge les résultats partiels d'une analyse précédente.

    Permet la reprise automatique si le script a été interrompu.
    Le JSON existant est lu et les résultats déjà calculés sont retournés.

    Args:
        chemin : Chemin vers le fichier JSON de sortie.

    Returns:
        Liste de résultats existants (vide si le fichier n'existe pas).
    """
    path = Path(chemin)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Le JSON de sortie contient une clé "paragraphes" avec les résultats
        existants = data.get("paragraphes", [])
        if existants:
            print(f"  Reprise détectée : {len(existants)} paragraphe(s) déjà traité(s).")
        return existants
    except (json.JSONDecodeError, KeyError):
        # Fichier corrompu → on repart de zéro
        print("  ⚠️  Fichier JSON existant illisible — reprise depuis le début.")
        return []


def sauvegarder_resultats(
    resultats: list[dict],
    chemin: str,
    meta_run: dict,
):
    """
    Sauvegarde les résultats dans le fichier JSON de sortie.

    Format du JSON :
    {
        "run": {métadonnées du traitement},
        "paragraphes": [liste des résultats par paragraphe]
    }

    La clé "run" permet à 08_visualise.py de connaître les conditions
    du traitement (manuscrit source, date, modèles utilisés, etc.).

    Args:
        resultats  : Liste des dicts résultats par paragraphe.
        chemin     : Chemin du fichier JSON de sortie.
        meta_run   : Métadonnées du traitement en cours.
    """
    path = Path(chemin)
    path.parent.mkdir(parents=True, exist_ok=True)

    sortie = {
        "run":         meta_run,
        "paragraphes": resultats,
    }

    path.write_text(
        json.dumps(sortie, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def generer_rapport_markdown(resultats: list[dict], chemin_json: str) -> Path:
    """
    Génère un rapport Markdown lisible à partir des résultats d'enrichissement.

    Format par paragraphe :
        ## § N — [début du texte]
        > [extrait du paragraphe]
        **Score de densité :** 0.XX

        ### Sources mobilisables
        **📄 source.pdf, p. N** — `citation | note | référence`
        [apport avec citation intégrée]

        ### Manques identifiés
        - [piste 1]

        ### Synthèse
        [2-3 phrases]

    Args:
        resultats   : Liste des dicts résultats par paragraphe.
        chemin_json : Chemin du JSON (sert à dériver le chemin .md).

    Returns:
        Chemin du fichier Markdown produit.
    """
    chemin_md = Path(chemin_json).with_suffix(".md")

    lignes = [
        "# Rapport d'enrichissement — Cartographie du manuscrit",
        f"*Généré le {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        f"*{len(resultats)} paragraphes analysés*",
        "",
        "---",
        "",
    ]

    for r in resultats:
        index = r.get("index", "?")
        texte = r.get("texte", "")
        score = r.get("score", 0.0)

        # En-tête
        debut = texte[:80].replace("\n", " ")
        lignes.append(f"## § {index} — {debut}…")
        lignes.append("")

        # Extrait
        apercu = texte[:300].replace("\n", " ")
        lignes.append(f"> {apercu}…")
        lignes.append("")

        # Score
        if score >= 0.7:
            indicateur = "🟢"
        elif score >= 0.4:
            indicateur = "🟡"
        else:
            indicateur = "🔴"
        lignes.append(f"**Score de densité documentaire :** {indicateur} {score:.2f}")
        lignes.append("")

        # Passages enrichis
        passages_enrichis = r.get("passages_enrichis", [])
        if passages_enrichis:
            lignes.append("### Sources mobilisables")
            lignes.append("")
            for p in passages_enrichis:
                usage = p.get("usage", "").strip()
                lignes.append(f"**📄 {p['source']}, p. {p['page']}**" +
                               (f" — `{usage}`" if usage else ""))
                apport = p.get("apport", "").strip()
                if apport:
                    lignes.append("")
                    lignes.append(apport)
                lignes.append("")
        else:
            lignes.append("*Aucun extrait directement mobilisable trouvé.*")
            lignes.append("")

        # Manques
        manques = r.get("manques", [])
        if manques:
            lignes.append("### Manques identifiés")
            lignes.append("")
            for m in manques:
                if m:
                    lignes.append(f"- {m}")
            lignes.append("")

        # Synthèse
        synthese = r.get("synthese", "").strip()
        if synthese:
            lignes.append("### Synthèse")
            lignes.append("")
            lignes.append(synthese)
            lignes.append("")

        lignes.append("---")
        lignes.append("")

    chemin_md.write_text("\n".join(lignes), encoding="utf-8")
    return chemin_md


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    # -------------------------------------------------------------------------
    # Parsing des arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="06_map_enrich.py — cartographie d'enrichissement du manuscrit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 06_map_enrich.py --manuscrit mon_manuscrit.txt
  python 06_map_enrich.py --manuscrit mon_manuscrit.txt --sortie resultats/enrich.json
  python 06_map_enrich.py --manuscrit mon_manuscrit.txt --top_k 6 --pause 0.0
        """
    )
    parser.add_argument(
        "--manuscrit",
        required=True,
        metavar="FICHIER",
        help="Fichier .txt du manuscrit complet (obligatoire)",
    )
    parser.add_argument(
        "--sortie",
        default=SORTIE_JSON,
        metavar="FICHIER",
        help=f"Fichier JSON de sortie (défaut : {SORTIE_JSON})",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        metavar="N",
        help="Nombre de passages FAISS par paragraphe",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=PAUSE_ENTRE_APPELS,
        metavar="SEC",
        help=f"Pause entre appels LLM en secondes (défaut : {PAUSE_ENTRE_APPELS})",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=PARAGRAPHE_MIN_CHARS,
        metavar="N",
        help=f"Longueur minimale d'un paragraphe (défaut : {PARAGRAPHE_MIN_CHARS})",
    )
    args = parser.parse_args()

    # Résolution de TOP_K : argument CLI > TOP_K_LOCAL > TOP_K de 00_config.py
    top_k = args.top_k or TOP_K_LOCAL or TOP_K

    # -------------------------------------------------------------------------
    # En-tête
    # -------------------------------------------------------------------------
    print("=" * 60)
    print("  06 — CARTOGRAPHIE D'ENRICHISSEMENT")
    print("=" * 60)
    print(f"  Manuscrit  : {args.manuscrit}")
    print(f"  Sortie     : {args.sortie}")
    print(f"  TOP_K      : {top_k}")
    print(f"  Pause LLM  : {args.pause}s")
    print(f"  Min chars  : {args.min_chars}")
    print(f"  Langue     : {LANGUE_PROMPTS}")
    print()

    # -------------------------------------------------------------------------
    # Chargement des ressources
    # -------------------------------------------------------------------------
    try:
        print("Chargement du manuscrit…")
        texte_manuscrit = charger_manuscrit(args.manuscrit)

        print("Segmentation en paragraphes…")
        paragraphes = segmenter_manuscrit(texte_manuscrit, args.min_chars)
        print(f"  {len(paragraphes)} paragraphe(s) exploitable(s) identifié(s).")

        print("Chargement de l'index FAISS…")
        for path in (index_file, metadata_file):
            if not path.exists():
                raise FileNotFoundError(
                    f"Fichier introuvable : {path.resolve()}\n"
                    "Lancez d'abord 03_build_embeddings.py."
                )
        index    = faiss.read_index(str(index_file))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        print(f"  {index.ntotal} vecteurs dans l'index.")

        print("Chargement du corpus…")
        corpus_text = charger_corpus()

        print("Initialisation du client LLM…")
        llm = LLMClient()

        print("Initialisation du client OpenAI (embeddings)…")
        client_openai = OpenAI()

    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Reprise éventuelle
    # -------------------------------------------------------------------------
    resultats = charger_resultats_existants(args.sortie)
    indices_traites = {r["index"] for r in resultats}

    # Paragraphes restants à traiter
    a_traiter = [p for p in paragraphes if p["index"] not in indices_traites]
    print(f"\n{len(a_traiter)} paragraphe(s) à analyser "
          f"({len(indices_traites)} déjà traité(s)).\n")

    if not a_traiter:
        print("✓ Tous les paragraphes ont déjà été traités.")
        print(f"  Résultats disponibles : {args.sortie}")
        print("  Lancez 08_visualise.py pour générer la visualisation.")
        sys.exit(0)

    # -------------------------------------------------------------------------
    # Métadonnées du run (pour traçabilité dans le JSON)
    # -------------------------------------------------------------------------
    meta_run = {
        "mode":             "enrichissement",
        "manuscrit":        args.manuscrit,
        "date_debut":       datetime.now().isoformat(),
        "top_k":            top_k,
        "langue":           LANGUE_PROMPTS,
        "nb_paragraphes":   len(paragraphes),
        "paragraphe_min_chars": args.min_chars,
    }

    # -------------------------------------------------------------------------
    # Boucle principale — traitement paragraphe par paragraphe
    # -------------------------------------------------------------------------
    print("─" * 60)
    print("  ANALYSE EN COURS")
    print("─" * 60)

    for i, para in enumerate(a_traiter):
        print(f"\n[{para['index']}/{len(paragraphes)}] "
              f"{para['nb_chars']} caractères")
        print(f"  Début : {para['texte'][:80].replace(chr(10), ' ')}…")

        # Étape 1 : embedding du paragraphe
        try:
            vecteur = encoder_texte(para["texte"], client_openai)
        except Exception as e:
            print(f"  ⚠️  Erreur embedding : {e} — paragraphe ignoré.")
            continue

        # Étape 2 : recherche FAISS
        passages = rechercher_passages(
            index, metadata, vecteur, corpus_text, top_k
        )
        print(f"  {len(passages)} passage(s) trouvé(s).")

        # Étape 3 : analyse LLM
        try:
            resultat = analyser_paragraphe(para, passages, llm, top_k)
        except Exception as e:
            print(f"  ⚠️  Erreur LLM : {e} — paragraphe ignoré.")
            continue

        print(f"  Score : {resultat['score']:.2f} / 1.0")

        # Ajout aux résultats et tri par index pour cohérence
        resultats.append(resultat)
        resultats.sort(key=lambda r: r["index"])

        # Sauvegarde intermédiaire tous les N paragraphes
        if (i + 1) % SAUVEGARDER_TOUS_LES_N == 0:
            sauvegarder_resultats(resultats, args.sortie, meta_run)
            print(f"  💾 Sauvegarde intermédiaire ({len(resultats)} paragraphes)")

        # Pause entre appels LLM
        if args.pause > 0 and i < len(a_traiter) - 1:
            time.sleep(args.pause)

    # -------------------------------------------------------------------------
    # Sauvegarde finale
    # -------------------------------------------------------------------------
    meta_run["date_fin"]         = datetime.now().isoformat()
    meta_run["nb_analyses"]      = len(resultats)
    meta_run["score_moyen"]      = round(
        sum(r["score"] for r in resultats) / len(resultats), 3
    ) if resultats else 0.0

    sauvegarder_resultats(resultats, args.sortie, meta_run)

    # Rapport Markdown lisible en parallèle du JSON
    chemin_md = generer_rapport_markdown(resultats, args.sortie)
    print(f"\n  📝 Rapport Markdown : {chemin_md.resolve()}")

    # -------------------------------------------------------------------------
    # Bilan
    # -------------------------------------------------------------------------
    scores = [r["score"] for r in resultats]
    score_moyen = sum(scores) / len(scores) if scores else 0.0
    zones_faibles = sum(1 for s in scores if s < 0.4)
    zones_fortes  = sum(1 for s in scores if s >= 0.7)

    print(f"\n{'═' * 60}")
    print(f"  BILAN")
    print(f"{'═' * 60}")
    print(f"  Paragraphes analysés : {len(resultats)}")
    print(f"  Score moyen          : {score_moyen:.2f} / 1.0")
    print(f"  Zones fortes (≥0.7)  : {zones_fortes} paragraphe(s)")
    print(f"  Zones faibles (<0.4) : {zones_faibles} paragraphe(s)")
    print(f"  Résultats sauvegardés : {Path(args.sortie).resolve()}")
    print(f"{'═' * 60}")
    print(f"\n  Étape suivante : python 08_visualise.py --source {args.sortie}")


if __name__ == "__main__":
    main()
