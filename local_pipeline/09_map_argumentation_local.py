"""
09_map_argumentation_local.py
=======================
Étape 9 du pipeline RAG — Analyse rhétorico-logique du manuscrit.

Rôle : Pour chaque section du manuscrit (issue du fichier map produit par
06_map_enrich_local.py), analyse la structure argumentative selon trois
cadres théoriques complémentaires :

    1. TOULMIN (1958) — Structure logique de l'argument
       Stephen Toulmin, The Uses of Argument (Cambridge UP, 1958).
       Six composantes : claim | grounds | warrant | backing | qualifier | rebuttal

    2. ADAM (1992) — Séquences argumentatives textuelles
       Jean-Michel Adam, Les textes : types et prototypes (Nathan, 1992).
       Quatre moments : thèse antérieure | données/argument | conclusion/thèse | restriction

    3. WALTON (1996) — Schèmes argumentatifs et sophismes
       Douglas Walton, Argumentation Schemes for Presumptive Reasoning (Erlbaum, 1996).
       Schèmes détectés : argument d'autorité | par analogie | causal | pragmatique
                          | pente glissante | généralisation hâtive | ad hominem | homme de paille

POSITION DANS LE PIPELINE :
    06_map_enrich_local.py   → map_{timestamp}.json  (passages corpus par section)
    09_map_argumentation_local.py  → argumentation_{timestamp}.json  (ce script)
    10_map_perelman_local.py       → perelman_{timestamp}.json

Ce script NE fait PAS de recherche vectorielle propre : il reprend le
fichier map produit par 06 pour disposer des passages corpus associés
à chaque section, ce qui lui permet de confronter la structure argumentative
du manuscrit avec les ressources du corpus.

SCORES PRODUITS (tous entre 0.0 et 1.0) :

    score_completude_toulmin [0–1]
        Mesure dans quelle proportion les six composantes de Toulmin sont
        présentes ou inférables dans le paragraphe.
        0.0 = seule la claim est présente, sans grounds ni warrant.
        1.0 = les six composantes sont identifiables.
        ⚠ Limite : un texte historique dense peut avoir toutes les composantes
        sans être pour autant convaincant ; la complétude est nécessaire mais
        non suffisante.

    score_coherence_warrant [0–1]
        Évalue la solidité du lien entre grounds (données) et claim (thèse) :
        le warrant (loi de passage) est-il explicite, implicite-plausible,
        ou absent/fragile ?
        0.0 = le lien données→thèse est arbitraire ou non justifié.
        1.0 = le warrant est explicitement formulé et étayé par le backing.
        ⚠ Limite : le LLM évalue la cohérence formelle, pas la vérité
        historiographique du warrant.

    score_risque_sophisme [0–1]
        Détecte la présence de schèmes argumentatifs fallacieux (au sens
        de Walton) : pente glissante, généralisation hâtive, appel à
        l'autorité non qualifiée, homme de paille, ad hominem.
        0.0 = aucun sophisme détecté.
        1.0 = le paragraphe repose principalement sur des arguments fallacieux.
        ⚠ Limite : certains schèmes (appel à l'autorité) sont légitimes
        dans certains contextes historiographiques. Le LLM ne distingue pas
        toujours un appel à l'autorité légitime d'un sophisme.

    score_charge_probatoire [0–1]
        Mesure si le paragraphe assume sa charge de preuve : les affirmations
        sont-elles étayées par des grounds explicites (sources, données,
        exemples) ou reposent-elles sur des postulats implicites ?
        0.0 = affirmations non étayées, postulats non déclarés.
        1.0 = chaque claim est accompagnée de grounds identifiables.
        ⚠ Limite : ce score ne mesure pas la qualité des sources, seulement
        leur présence formelle dans l'argumentation.

    score_robustesse_globale [0–1]
        Score synthétique pondéré :
            score_robustesse_globale = (
                score_completude_toulmin  × 0.20
              + score_coherence_warrant   × 0.35
              + score_charge_probatoire   × 0.25
              + (1 − score_risque_sophisme) × 0.20
            )
        La pénalisation des sophismes (1 − score) est délibérée : un argument
        sophistique réduit la robustesse globale même si la structure formelle
        est complète. Le warrant est le facteur le plus lourd (0.35) car c'est
        le maillon le plus souvent absent dans l'écriture historienne.

    schema_walton_dominant [str]
        Le schème argumentatif le plus représenté dans la section :
        "autorité" | "analogie" | "causal" | "pragmatique" |
        "pente_glissante" | "generalisation" | "ad_hominem" |
        "homme_de_paille" | "aucun"

SORTIE JSON (trois niveaux, cohérente avec 07) :
    {
      "script"         : "09_map_argumentation",
      "timestamp"      : "…",
      "modele_llm"     : "…",
      "score_global"   : 0.72,
      "profil_dominant": "causal",
      "sections": [
        {
          "id"                      : 1,
          "titre"                   : "…",
          "score_completude_toulmin": 0.80,
          "score_coherence_warrant" : 0.65,
          "score_risque_sophisme"   : 0.10,
          "score_charge_probatoire" : 0.75,
          "score_robustesse_globale": 0.74,
          "schema_walton_dominant"  : "causal",
          "composantes_toulmin": {
              "claim"   : "…",
              "grounds" : "…",
              "warrant" : "…",
              "backing" : "…",
              "qualifier": "…",
              "rebuttal" : "…"
          },
          "sequence_adam": {
              "these_anterieure": "…",
              "donnees"         : "…",
              "conclusion"      : "…",
              "restriction"     : "…"
          },
          "sophismes_detectes": ["…"],
          "analyse"             : "… texte brut LLM …",
          "passages_corpus"     : [{"source": "…", "page": 0, "extrait": "…"}]
        }
      ]
    }

CADRE THÉORIQUE — NOTE POUR L'HISTORIEN :

    Toulmin s'applique à l'unité argumentative (le paragraphe ou la séquence).
    Adam opère à l'échelle de la séquence textuelle (souvent plusieurs paragraphes).
    Walton cible les schèmes récurrents — il est le seul des trois à distinguer
    explicitement arguments légitimes et fallacieux.

    Ces trois cadres ne sont pas redondants : Toulmin dit SI l'argument est
    complet ; Walton dit QUEL TYPE d'argument est utilisé et SI il est risqué ;
    Adam dit comment l'argument s'inscrit dans la progression du texte.

    La confrontation avec le corpus (passages_corpus issus de 06) permet de
    vérifier si les grounds invoqués dans le manuscrit sont effectivement
    attestés dans la littérature secondaire indexée.

UTILISATION :
    python 09_map_argumentation_local.py
    python 09_map_argumentation_local.py --map_file outputs/map_20240412_143022.json
    python 09_map_argumentation_local.py --sections 3 7 12

Environnement conda : rag_historien
Dépendances : ollama (serveur actif), requests, json, pathlib
"""

# =============================================================================
# PARAMÈTRES — modifier ici sans toucher au reste du script
# =============================================================================

# LLM_MODEL et OLLAMA_URL sont lus depuis rag_config_local.py.
# Pour changer de modèle sur tout le pipeline local, modifier LLM_MODEL
# dans rag_config_local.py — voir le tableau des modèles disponibles.
# Recommandation pour ce script : deepseek-r1:14b ou gemma3:12b
# (meilleure représentation des cadres Toulmin/Adam/Walton).
from rag_config_local import LLM_MODEL, OLLAMA_URL

# Température basse pour une analyse structurée et reproductible.
TEMPERATURE = 0.1

# Nombre maximum de tokens en sortie par section.
# L'analyse Toulmin + Adam + Walton est verbeuse : 2500 est un minimum sûr.
MAX_TOKENS = 2500

# Nombre maximum de passages corpus inclus dans chaque prompt.
MAX_PASSAGES_PAR_SECTION = 6

# Répertoires.
MAP_DIR    = "outputs"
OUTPUT_DIR = "outputs"

# Pause en secondes entre deux appels Ollama (évite la surchauffe sur M2).
PAUSE_INTER_SECTION = 1.0

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# =============================================================================
# SCHÈMES WALTON — référentiel interne
# =============================================================================

# Liste des schèmes argumentatifs de Walton reconnus par ce script.
# Les cinq premiers sont des schèmes légitimes, les quatre derniers sont
# des schèmes fallacieux (sophismes présomptifs).
SCHEMES_WALTON = {
    # ── Schèmes légitimes ──────────────────────────────────────────────────
    "autorité"     : "Argument from Expert Opinion — la thèse est étayée par "
                     "l'autorité d'un expert ou d'une institution reconnue.",
    "analogie"     : "Argument from Analogy — la thèse est soutenue par une "
                     "comparaison avec un cas similaire.",
    "causal"       : "Argument from Cause to Effect — la thèse découle d'une "
                     "relation causale explicitée.",
    "pragmatique"  : "Pragmatic Argument — la thèse est justifiée par ses "
                     "conséquences pratiques ou ses effets attendus.",
    # ── Schèmes fallacieux ─────────────────────────────────────────────────
    "pente_glissante"  : "Slippery Slope — enchaînement causal non étayé qui "
                         "conduit à une conséquence extrême.",
    "generalisation"   : "Hasty Generalization — conclusion générale tirée d'un "
                         "échantillon trop limité ou non représentatif.",
    "ad_hominem"       : "Ad Hominem — attaque de la personne plutôt que de "
                         "l'argument.",
    "homme_de_paille"  : "Straw Man — déformation de la position adverse pour "
                         "la réfuter plus facilement.",
}

SCHEMES_FALLACIEUX = {"pente_glissante", "generalisation", "ad_hominem", "homme_de_paille"}

# =============================================================================
# FONCTIONS — chargement des données
# =============================================================================

def trouver_dernier_map(map_dir: Path) -> Path:
    """
    Recherche le fichier map_{timestamp}.json le plus récent dans map_dir.
    Lève FileNotFoundError si aucun fichier n'est trouvé.
    """
    fichiers = sorted(map_dir.glob("map_*.json"), reverse=True)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier map_*.json trouvé dans {map_dir}.\n"
            "Lancez d'abord 06_map_enrich_local.py."
        )
    return fichiers[0]


def charger_map(chemin: Path) -> dict:
    """Charge et valide le fichier map JSON produit par 06."""
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    if "sections" not in data:
        raise ValueError(f"Le fichier {chemin.name} ne contient pas de clé 'sections'.")
    return data


# =============================================================================
# FONCTIONS — construction du prompt
# =============================================================================

def formater_passages(passages: list[dict], max_passages: int) -> str:
    """
    Formate les passages corpus pour inclusion dans le prompt.
    Chaque passage est identifié par sa source et sa page.
    Tronque à max_passages pour maîtriser la taille du prompt.
    """
    lignes = []
    for i, p in enumerate(passages[:max_passages], 1):
        source = p.get("source", "source inconnue")
        page   = p.get("page", "?")
        texte  = p.get("extrait", p.get("texte", "")).strip()
        if texte:
            lignes.append(f"[Passage {i} — {source}, p. {page}]\n{texte}")
    return "\n\n".join(lignes) if lignes else "(aucun passage corpus disponible)"


def construire_prompt(titre: str, texte: str, passages_str: str) -> str:
    """
    Construit le prompt d'analyse argumentative pour une section.

    Le prompt demande au LLM trois analyses distinctes (Toulmin, Adam, Walton)
    puis les cinq scores dans un format parsable par regex.
    La langue du prompt est le français pour cohérence avec le projet,
    mais le LLM doit analyser le texte quelle que soit sa langue (FR/EN/DE/IT).
    """
    return f"""Tu es un expert en rhétorique et logique argumentative, spécialisé dans
les textes académiques et historiques. Analyse le passage suivant selon
trois cadres théoriques, puis attribue les scores demandés.

══════════════════════════════════════════
SECTION DU MANUSCRIT : {titre}
══════════════════════════════════════════
{texte}

══════════════════════════════════════════
PASSAGES DU CORPUS (littérature secondaire)
══════════════════════════════════════════
{passages_str}

══════════════════════════════════════════
ANALYSE DEMANDÉE
══════════════════════════════════════════

━━━ 1. ANALYSE TOULMIN ━━━
Identifie les six composantes de l'argument (si absentes, indique ABSENT) :
  CLAIM     : la thèse ou conclusion principale du passage
  GROUNDS   : les données, faits ou preuves invoquées
  WARRANT   : la règle ou principe qui autorise le passage de GROUNDS à CLAIM
  BACKING   : l'autorité ou la preuve qui légitime le WARRANT
  QUALIFIER : les nuances, limites ou modalités de la CLAIM ("probablement", "dans certains cas"…)
  REBUTTAL  : les cas d'exception ou objections anticipées

━━━ 2. ANALYSE ADAM ━━━
Identifie les quatre moments de la séquence argumentative :
  THÈSE ANTÉRIEURE : la position de départ ou la thèse adverse implicite
  DONNÉES          : les arguments et preuves mobilisés
  CONCLUSION/THÈSE : la nouvelle thèse défendue
  RESTRICTION      : les concessions ou limites apportées à la conclusion

━━━ 3. ANALYSE WALTON — SCHÈMES ARGUMENTATIFS ━━━
Identifie le(s) schème(s) argumentatif(s) présents dans le passage parmi :
  Légitimes  : autorité | analogie | causal | pragmatique
  Fallacieux : pente_glissante | generalisation | ad_hominem | homme_de_paille

Pour chaque schème détecté, cite le passage exact du texte qui l'illustre.
Si plusieurs schèmes coexistent, classe-les par ordre d'importance.
Schème dominant (un seul mot, parmi la liste ci-dessus ou "aucun") :
  SCHEMA_DOMINANT: <schème>

Sophismes détectés (liste vide si aucun) :
  SOPHISMES: <schème1>, <schème2>  (ou AUCUN)

━━━ 4. CONFRONTATION AVEC LE CORPUS ━━━
Les GROUNDS invoqués dans le manuscrit sont-ils attestés dans les passages corpus ?
Pour chaque ground identifié : ATTESTÉ | ABSENT DU CORPUS | CONTREDIT PAR LE CORPUS

━━━ 5. SCORES (format strict, ne pas modifier) ━━━
Évalue chaque dimension sur 10 (entier ou décimal, ex : 7 ou 7.5) :
  SCORE_COMPLETUDE_TOULMIN: X/10   (proportion des 6 composantes présentes ou inférables)
  SCORE_COHERENCE_WARRANT: X/10   (solidité du lien grounds→claim via le warrant)
  SCORE_RISQUE_SOPHISME: X/10     (0 = aucun sophisme | 10 = argument principalement fallacieux)
  SCORE_CHARGE_PROBATOIRE: X/10   (chaque claim est-elle étayée par des grounds explicites ?)
  SCORE_ROBUSTESSE_GLOBALE: X/10  (synthèse pondérée — calcule selon : complétude×0.20 + warrant×0.35 + charge×0.25 + (10−sophisme)×0.20, divise par 10)

━━━ 6. SYNTHÈSE ━━━
En 3–4 phrases : évaluation argumentative globale du passage.
Quel est le principal point fort ? Le principal point faible ?
Quelle révision prioritaire suggères-tu à l'historien ?"""


# =============================================================================
# FONCTIONS — appel LLM
# =============================================================================

def appel_ollama(prompt: str) -> str:
    """
    Envoie le prompt à Ollama et retourne la réponse brute.

    Gestion des erreurs :
    - Timeout à 120 secondes (l'analyse Toulmin est longue).
    - En cas d'échec réseau : retourne une chaîne vide et affiche un warning.
    - Le champ "done" est vérifié pour s'assurer que la réponse est complète.
    """
    payload = {
        "model"  : LLM_MODEL,
        "prompt" : prompt,
        "stream" : False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("done", False):
            print("  ⚠ Réponse Ollama incomplète (done=False).", file=sys.stderr)
        return data.get("response", "")
    except requests.exceptions.ConnectionError:
        print(
            "  ❌ Impossible de joindre Ollama. Vérifiez que le serveur est actif : ollama serve",
            file=sys.stderr,
        )
        return ""
    except requests.exceptions.Timeout:
        print("  ❌ Timeout Ollama (>120 s). Section ignorée.", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  ❌ Erreur Ollama : {e}", file=sys.stderr)
        return ""


# =============================================================================
# FONCTIONS — extraction des scores et composantes
# =============================================================================

# Patterns regex pour l'extraction des scores depuis la réponse LLM.
# Tolèrent les espaces variés et les valeurs décimales (7.5/10).
# Le /10 est rendu optionnel (?:...) : qwen2.5 omet parfois le dénominateur.
# Le score est normalisé vers [0,1] par _normaliser_score() selon sa valeur.
PATTERNS_SCORES = {
    "score_completude_toulmin": r"SCORE_COMPLETUDE_TOULMIN\s*:\s*(\d+(?:[.,]\d+)?)(?:\s*/\s*10)?",
    "score_coherence_warrant" : r"SCORE_COHERENCE_WARRANT\s*:\s*(\d+(?:[.,]\d+)?)(?:\s*/\s*10)?",
    "score_risque_sophisme"   : r"SCORE_RISQUE_SOPHISME\s*:\s*(\d+(?:[.,]\d+)?)(?:\s*/\s*10)?",
    "score_charge_probatoire" : r"SCORE_CHARGE_PROBATOIRE\s*:\s*(\d+(?:[.,]\d+)?)(?:\s*/\s*10)?",
    "score_robustesse_globale": r"SCORE_ROBUSTESSE_GLOBALE\s*:\s*(\d+(?:[.,]\d+)?)(?:\s*/\s*10)?",
}

# Noms courts alternatifs acceptés par les niveaux 3 et 4
NOMS_COURTS_SCORES = {
    "score_completude_toulmin": ["complétude", "completude", "complt", "completude_toulmin"],
    "score_coherence_warrant" : ["cohérence", "coherence", "warrant", "warrt"],
    "score_risque_sophisme"   : ["sophisme", "sophistique", "sophis", "risque_sophisme"],
    "score_charge_probatoire" : ["probatoire", "probat", "charge_probatoire"],
    "score_robustesse_globale": ["robustesse", "robust", "robustesse_globale"],
}

VALEUR_NEUTRE = 0.5  # Valeur de repli si un score n'est pas extrait.


def _normaliser_score(valeur: float, avec_denominateur: bool) -> float:
    """Normalise un score brut vers [0.0, 1.0]."""
    if avec_denominateur or valeur > 1.0:
        valeur = valeur / 10.0
    return round(max(0.0, min(1.0, valeur)), 3)


def extraire_scores(texte_llm: str) -> dict:
    """
    Extrait les cinq scores Toulmin depuis la réponse LLM brute.

    Quatre niveaux de robustesse, du plus strict au plus permissif :

    Niveau 1 — Format strict : SCORE_X: N/10
        Accepte point ou virgule comme séparateur décimal.
    Niveau 2 — Espaces autour du / : SCORE_X: N / 10
        (couvert par le même pattern que niveau 1)
    Niveau 3 — Nom court sans préfixe SCORE_
        Ex : "complétude : 8" ou "robustesse = 7.5"
    Niveau 4 — Tableau Markdown
        | label | valeur |

    Si aucun niveau ne trouve le score : retourne VALEUR_NEUTRE (0.5).
    """
    scores = {}

    for nom, pattern in PATTERNS_SCORES.items():
        # Niveau 1 + 2 — format strict
        m = re.search(pattern, texte_llm, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(",", "."))
            # Détecte si /10 est présent dans la correspondance complète
            avec_denom = bool(re.search(r"/\s*10", m.group(0)))
            # Si pas de /10 mais valeur > 1 → forcément sur 10
            if not avec_denom and v > 1.0:
                avec_denom = True
            scores[nom] = _normaliser_score(v, avec_denominateur=avec_denom)
            continue

        # Niveau 3 — nom court avec ou sans /10
        trouve = False
        for nom_court in NOMS_COURTS_SCORES.get(nom, []):
            pat3 = (rf"(?:^|[\s|])(?:{re.escape(nom_court)})"
                    rf"\s*[:=]\s*(\d+(?:[.,]\d+)?)\s*(?:/\s*10)?")
            m3 = re.search(pat3, texte_llm, re.IGNORECASE | re.MULTILINE)
            if m3:
                v = float(m3.group(1).replace(",", "."))
                avec_denom = bool(re.search(
                    r"/\s*10", texte_llm[m3.start():m3.end() + 10]
                ))
                scores[nom] = _normaliser_score(v, avec_denominateur=avec_denom)
                trouve = True
                break

        if trouve:
            continue

        # Niveau 4 — tableau Markdown
        for nom_court in NOMS_COURTS_SCORES.get(nom, []):
            pat4 = (rf"\|\s*[^|]*{re.escape(nom_court)}[^|]*\|"
                    rf"\s*(\d+(?:[.,]\d+)?)\s*\|")
            m4 = re.search(pat4, texte_llm, re.IGNORECASE)
            if m4:
                v = float(m4.group(1).replace(",", "."))
                scores[nom] = _normaliser_score(v, avec_denominateur=False)
                break
        else:
            print(f"  ⚠ Score non trouvé : {nom} → {VALEUR_NEUTRE}", file=sys.stderr)
            scores[nom] = VALEUR_NEUTRE

    return scores


def extraire_schema_walton(texte_llm: str) -> tuple[str, list[str]]:
    """
    Extrait le schème Walton dominant et la liste des sophismes détectés.

    Retourne :
        schema_dominant : str (un nom de schème ou "aucun")
        sophismes       : list[str] (noms des schèmes fallacieux détectés)
    """
    # Schème dominant
    m_schema = re.search(
        r"SCHEMA_DOMINANT\s*:\s*(\w+)", texte_llm, re.IGNORECASE
    )
    schema_dominant = m_schema.group(1).lower() if m_schema else "aucun"
    # Validation : le schème doit être dans le référentiel
    if schema_dominant not in SCHEMES_WALTON and schema_dominant != "aucun":
        schema_dominant = "aucun"

    # Sophismes détectés
    m_soph = re.search(
        r"SOPHISMES\s*:\s*(.+)", texte_llm, re.IGNORECASE
    )
    sophismes = []
    if m_soph:
        raw = m_soph.group(1).strip()
        if raw.upper() != "AUCUN":
            candidats = [s.strip().lower() for s in re.split(r"[,;]", raw)]
            sophismes = [s for s in candidats if s in SCHEMES_FALLACIEUX]

    return schema_dominant, sophismes


def extraire_composantes_toulmin(texte_llm: str) -> dict:
    """
    Extrait les six composantes Toulmin depuis la réponse LLM.
    Les composantes absentes reçoivent la valeur "ABSENT".
    """
    composantes = {}
    for comp in ["CLAIM", "GROUNDS", "WARRANT", "BACKING", "QUALIFIER", "REBUTTAL"]:
        # Cherche "CLAIM : texte" jusqu'à la prochaine composante ou fin de ligne
        m = re.search(
            rf"{comp}\s*:\s*(.+?)(?=\n\s*(?:CLAIM|GROUNDS|WARRANT|BACKING|QUALIFIER|REBUTTAL|━|$))",
            texte_llm,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            valeur = m.group(1).strip()
            composantes[comp.lower()] = valeur if valeur.upper() != "ABSENT" else "ABSENT"
        else:
            composantes[comp.lower()] = "ABSENT"
    return composantes


def extraire_sequence_adam(texte_llm: str) -> dict:
    """
    Extrait les quatre moments de la séquence Adam.
    """
    sequence = {}
    champs = {
        "these_anterieure": r"THÈSE ANTÉRIEURE\s*:\s*(.+?)(?=\n\s*(?:DONNÉES|CONCLUSION|RESTRICTION|━|$))",
        "donnees"         : r"DONNÉES\s*:\s*(.+?)(?=\n\s*(?:CONCLUSION|RESTRICTION|━|$))",
        "conclusion"      : r"CONCLUSION(?:/THÈSE)?\s*:\s*(.+?)(?=\n\s*(?:RESTRICTION|━|$))",
        "restriction"     : r"RESTRICTION\s*:\s*(.+?)(?=\n\s*(?:━|$))",
    }
    for cle, pattern in champs.items():
        m = re.search(pattern, texte_llm, re.IGNORECASE | re.DOTALL)
        sequence[cle] = m.group(1).strip() if m else "non identifié"
    return sequence


# =============================================================================
# FONCTION — calcul du score robustesse (vérification de cohérence)
# =============================================================================

def calculer_robustesse(scores: dict) -> float:
    """
    Recalcule le score_robustesse_globale selon la pondération documentée.
    Utilisé comme vérification si le LLM a fourni une valeur incohérente.

        score_robustesse = completude×0.20 + warrant×0.35
                         + charge×0.25 + (1−sophisme)×0.20

    Retourne un float [0.0, 1.0].
    """
    robustesse = (
        scores.get("score_completude_toulmin", 0.5) * 0.20
        + scores.get("score_coherence_warrant", 0.5) * 0.35
        + scores.get("score_charge_probatoire", 0.5) * 0.25
        + (1.0 - scores.get("score_risque_sophisme", 0.5)) * 0.20
    )
    return round(robustesse, 3)


# =============================================================================
# FONCTION — traitement d'une section
# =============================================================================

def analyser_section(section: dict) -> dict:
    """
    Orchestre l'analyse argumentative d'une section du manuscrit.

    Étapes :
        1. Récupère le texte et les passages corpus de la section.
        2. Construit le prompt.
        3. Appelle Ollama.
        4. Extrait les scores, composantes et schèmes.
        5. Recalcule la robustesse pour vérification.
        6. Retourne le dictionnaire de résultats.
    """
    titre  = section.get("titre", f"Section {section.get('id', '?')}")
    texte  = section.get("texte", "")
    passages = section.get("passages_corpus", [])

    if not texte.strip():
        print(f"  ⚠ Section sans texte — ignorée.", file=sys.stderr)
        return None

    passages_str = formater_passages(passages, MAX_PASSAGES_PAR_SECTION)
    prompt       = construire_prompt(titre, texte, passages_str)
    reponse_brute = appel_ollama(prompt)

    if not reponse_brute:
        return None

    # Extraction
    scores          = extraire_scores(reponse_brute)
    schema_dominant, sophismes = extraire_schema_walton(reponse_brute)
    composantes_toulmin = extraire_composantes_toulmin(reponse_brute)
    sequence_adam       = extraire_sequence_adam(reponse_brute)

    # Vérification et recalcul de la robustesse
    robustesse_verif = calculer_robustesse(scores)
    ecart = abs(scores.get("score_robustesse_globale", 0.5) - robustesse_verif)
    if ecart > 0.15:
        # Le LLM s'est trompé dans son calcul : on corrige silencieusement
        print(
            f"  ℹ Robustesse LLM corrigée : {scores['score_robustesse_globale']:.3f}"
            f" → {robustesse_verif:.3f} (écart {ecart:.3f})",
            file=sys.stderr,
        )
        scores["score_robustesse_globale"] = robustesse_verif

    return {
        "id"                       : section.get("id"),
        "titre"                    : titre,
        **scores,
        "schema_walton_dominant"   : schema_dominant,
        "sophismes_detectes"       : sophismes,
        "composantes_toulmin"      : composantes_toulmin,
        "sequence_adam"            : sequence_adam,
        "analyse"                  : reponse_brute,    # texte brut conservé
        "passages_corpus"          : [
            {
                "source" : p.get("source", ""),
                "page"   : p.get("page", ""),
                "extrait": p.get("extrait", p.get("texte", ""))[:300],
            }
            for p in passages[:MAX_PASSAGES_PAR_SECTION]
        ],
    }


# =============================================================================
# CALCUL DU SCORE GLOBAL
# =============================================================================

def calculer_score_global(sections_analysees: list[dict]) -> tuple[float, str]:
    """
    Calcule le score_robustesse_globale moyen sur toutes les sections analysées.
    Identifie le schème Walton dominant au niveau du manuscrit entier.

    Retourne : (score_global, schema_manuscrit_dominant)
    """
    if not sections_analysees:
        return 0.0, "aucun"

    robustesses = [s["score_robustesse_globale"] for s in sections_analysees]
    score_global = round(sum(robustesses) / len(robustesses), 3)

    # Schème dominant = le plus fréquent parmi les sections
    comptage = {}
    for s in sections_analysees:
        schema = s.get("schema_walton_dominant", "aucun")
        comptage[schema] = comptage.get(schema, 0) + 1
    schema_dominant = max(comptage, key=comptage.get)

    return score_global, schema_dominant


# =============================================================================
# SAUVEGARDE
# =============================================================================

def sauvegarder(
    sections_analysees: list[dict],
    score_global: float,
    profil_dominant: str,
    output_dir: Path,
) -> Path:
    """
    Sauvegarde les résultats dans argumentation_{timestamp}.json.
    Retourne le chemin du fichier créé.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin = output_dir / f"argumentation_{timestamp}.json"

    sortie = {
        "script"          : "09_map_argumentation",
        "timestamp"       : timestamp,
        "modele_llm"      : LLM_MODEL,
        "temperature"     : TEMPERATURE,
        "score_global"    : score_global,
        "profil_dominant" : profil_dominant,
        "nb_sections"     : len(sections_analysees),
        "sections"        : sections_analysees,
    }

    with open(chemin, "w", encoding="utf-8") as f:
        json.dump(sortie, f, ensure_ascii=False, indent=2)

    return chemin


# =============================================================================
# BILAN TERMINAL
# =============================================================================

def afficher_bilan(sections_analysees: list[dict], score_global: float) -> None:
    """
    Affiche un tableau récapitulatif des scores par section dans le terminal.
    Utilise des barres Unicode pour une lecture rapide.
    """
    print("\n" + "═" * 72)
    print(f"  BILAN — 09_map_argumentation | Score global : {score_global:.3f}")
    print("═" * 72)
    print(
        f"  {'§':>3}  {'Titre':<28}  {'Complt':>6}  {'Warrt':>6}  "
        f"{'Probat':>6}  {'Sophis':>6}  {'Robust':>6}  Schème"
    )
    print("─" * 72)

    for s in sections_analysees:
        bar_r = "█" * int(s["score_robustesse_globale"] * 10)
        print(
            f"  {s['id']:>3}  {s['titre'][:28]:<28}  "
            f"{s['score_completude_toulmin']:.2f}    "
            f"{s['score_coherence_warrant']:.2f}    "
            f"{s['score_charge_probatoire']:.2f}    "
            f"{s['score_risque_sophisme']:.2f}    "
            f"{s['score_robustesse_globale']:.2f}  "
            f"{s.get('schema_walton_dominant', 'aucun')}"
        )
        if s.get("sophismes_detectes"):
            print(f"       ⚠ Sophismes : {', '.join(s['sophismes_detectes'])}")

    print("═" * 72 + "\n")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="09_map_argumentation — Analyse Toulmin + Adam + Walton du manuscrit."
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default=None,
        help="Chemin vers le fichier map JSON produit par 06_map_enrich_local.py. "
             "Si absent, utilise le plus récent dans outputs/.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Identifiants des sections à analyser (ex : --sections 1 3 7). "
             "Si absent, toutes les sections sont analysées.",
    )
    args = parser.parse_args()

    map_dir    = Path(MAP_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chargement du fichier map
    if args.map_file:
        map_file = Path(args.map_file)
        if not map_file.is_absolute():
            map_file = map_dir / map_file
    else:
        print("Aucun fichier map spécifié — recherche du plus récent…")
        map_file = trouver_dernier_map(map_dir)

    print(f"Chargement : {map_file.name}")
    map_data = charger_map(map_file)
    sections = map_data.get("sections", [])

    # Filtre optionnel par identifiant de section
    if args.sections:
        sections = [s for s in sections if s.get("id") in args.sections]
        if not sections:
            print(f"❌ Aucune section trouvée avec les identifiants {args.sections}.")
            sys.exit(1)

    print(f"  {len(sections)} section(s) à analyser.")
    print(f"  Modèle LLM : {LLM_MODEL} | T° : {TEMPERATURE}\n")

    # Boucle d'analyse
    sections_analysees = []
    for sec in sections:
        print(f"§ {sec.get('id')} — {sec.get('titre', '?')}")
        resultat = analyser_section(sec)
        if resultat:
            r = resultat["score_robustesse_globale"]
            s = resultat.get("schema_walton_dominant", "aucun")
            print(f"  Robustesse : {r:.3f} | Schème dominant : {s}")
            sections_analysees.append(resultat)
        else:
            print("  ⚠ Section ignorée (réponse vide ou texte manquant).")
        time.sleep(PAUSE_INTER_SECTION)
        print()

    if not sections_analysees:
        print("❌ Aucune section analysée. Vérifiez le fichier map et Ollama.")
        sys.exit(1)

    # Score global et sauvegarde
    score_global, profil_dominant = calculer_score_global(sections_analysees)
    chemin_sortie = sauvegarder(sections_analysees, score_global, profil_dominant, output_dir)

    afficher_bilan(sections_analysees, score_global)
    print(f"✅ Analyse sauvegardée : {chemin_sortie.name}")
    print(f"   Score robustesse global : {score_global:.3f}")
    print(f"   Schème argumentatif dominant : {profil_dominant}\n")


if __name__ == "__main__":
    main()
