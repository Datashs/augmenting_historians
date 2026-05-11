"""
10_map_perelman_local.py
==================
Étape 10 du pipeline RAG — Analyse rhétorique selon la Nouvelle Rhétorique.

Rôle : Pour chaque section du manuscrit (issue du fichier map produit par
06_map_enrich_local.py), analyse les techniques argumentatives selon le
cadre de la Nouvelle Rhétorique de Chaïm Perelman et Lucie Olbrechts-Tyteca
(Traité de l'argumentation, 1958).

RÉFÉRENCE THÉORIQUE :
    Chaïm Perelman & Lucie Olbrechts-Tyteca,
    Traité de l'argumentation — La Nouvelle Rhétorique.
    Bruxelles : Éditions de l'Université de Bruxelles, 1958 (rééd. 1988).

POSITION DANS LE PIPELINE :
    06_map_enrich_local.py   → map_{timestamp}.json
    09_map_argumentation_local.py  → argumentation_{timestamp}.json
    10_map_perelman_local.py       → perelman_{timestamp}.json  (ce script)

Ce script NE fait PAS de recherche vectorielle propre : il reprend le
fichier map produit par 06 (passages corpus par section) et, optionnellement,
peut lire le fichier argumentation produit par 09 pour croiser les analyses.

CADRE THÉORIQUE — LA NOUVELLE RHÉTORIQUE :

    Perelman distingue deux grands types de techniques argumentatives :

    ── A. TECHNIQUES D'ASSOCIATION ──────────────────────────────────────────
    L'orateur rapproche des éléments séparés pour les faire paraître solidaires.

    A1. Arguments quasi-logiques
        Fondés sur une apparence de rigueur formelle (ressemblance avec
        la logique ou les mathématiques) :
        — incompatibilité : deux propositions ne peuvent être vraies ensemble
        — identité / définition : ramener une chose à une autre par définition
        — transitif : si A=B et B=C alors A=C
        — réciprocité : si A→B alors B→A (souvent abusive)
        — inclusion de la partie : le tout implique la partie
        — comparaison / sacrifice : peser deux choses sur une même échelle

    A2. Arguments fondés sur la structure du réel
        Fondés sur des liaisons reconnues dans le réel :
        — lien causal : cause → effet, moyen → fin, fin → moyen
        — argument pragmatique : juger un acte par ses conséquences
        — argument d'autorité / prestige : s'appuyer sur une personne ou
          institution jugée compétente
        — argument d'illustration : un exemple confirme une règle
        — argument de modèle / anti-modèle : comportement à imiter ou éviter
        — analogie : rapport A:B :: C:D

    A3. Arguments qui fondent la structure du réel
        L'orateur crée de nouvelles liaisons pour les faire admettre :
        — exemple (généralisation inductive) : d'un cas, tirer une règle
        — illustration (renforcement) : cas qui appuie une règle admise
        — métaphore : analogie condensée, souvent euphémisante
        — dissociation conceptuelle (voir ci-dessous)

    ── B. TECHNIQUES DE DISSOCIATION ────────────────────────────────────────
    L'orateur scinde un concept apparemment uni en deux termes hiérarchisés
    (paires philosophiques), le terme II dévalorisant le terme I :

        apparence / réalité          individu / collectif
        moyen / fin                  subjectif / objectif
        relatif / absolu             intérêt personnel / intérêt général
        lettre / esprit              acte / personne
        théorie / pratique

    La dissociation est une technique rhétorique puissante en histoire :
    elle permet de distinguer la "vraie" causalité de ses apparences,
    ou les "vraies" intentions d'un acteur de ses déclarations.

    ── C. LA NOTION D'AUDITOIRE ─────────────────────────────────────────────
    Perelman distingue :
        — l'auditoire universel : l'argumentation prétend s'adresser à
          tout être raisonnable (ambition démonstrative)
        — l'auditoire particulier : l'argumentation cible un groupe
          partageant des valeurs ou prémisses communes
    En histoire académique, l'auditoire est typiquement la communauté des
    historiens — un auditoire particulier avec des conventions épistémiques
    spécifiques (note de bas de page, critique des sources, etc.).

SCORES PRODUITS (tous entre 0.0 et 1.0) :

    score_force_persuasive [0–1]
        Évalue l'efficacité rhétorique globale du passage pour son auditoire
        académique : le texte exploite-t-il des techniques argumentatives
        adaptées aux conventions historiographiques ?
        0.0 = texte purement assertif, sans dispositif rhétorique.
        1.0 = argumentation sophistiquée, multiplicité des techniques bien
              articulées.
        ⚠ Limite : ce score mesure la sophistication rhétorique, pas la
        vérité ou la solidité logique de l'argument (voir 09 pour cela).

    score_ancrage_auditoire [0–1]
        Mesure dans quelle mesure l'argumentation est ancrée dans les prémisses
        partagées par l'auditoire académique des historiens (références aux
        débats historiographiques, conventions de preuve, normes de citation).
        0.0 = ignorance totale des codes disciplinaires.
        1.0 = parfaite maîtrise des conventions rhétoriques de la discipline.
        ⚠ Limite : dépend de la richesse des passages corpus disponibles ;
        si le corpus est lacunaire dans un sous-domaine, le score peut être
        sous-estimé.

    score_coherence_valeurs [0–1]
        Vérifie que les valeurs mobilisées par l'auteur (vérité, progrès,
        objectivité, représentativité, etc.) sont cohérentes à travers la
        section : l'auteur ne s'appuie pas sur des valeurs contradictoires
        selon les besoins de l'argument.
        0.0 = valeurs contradictoires ou opportunistes.
        1.0 = système de valeurs cohérent et explicite.
        ⚠ Limite : le LLM détecte les incohérences formelles, pas les
        présupposés épistémologiques profonds.

    score_risque_sophistique_rhethorique [0–1]
        Détecte les usages rhétoriques problématiques au sens de Perelman :
        — auditoire universel revendiqué abusivement (prétendre à l'évidence)
        — métaphore qui masque un glissement conceptuel
        — dissociation qui crée une fausse dichotomie
        — argument d'autorité sans qualification de l'autorité
        — pétition de principe camouflée en argument quasi-logique
        0.0 = aucun usage sophistique détecté.
        1.0 = argumentation reposant principalement sur des effets rhétoriques
              non fondés.
        ⚠ Limite : la frontière entre rhétorique efficace et sophisme
        rhétorique est contextuelle — le LLM peut hésiter sur des cas limites.

    score_profil_argumentatif [0–1]
        Score composite représentant l'équilibre entre sophistication
        rhétorique et risque sophistique :
            score_profil = (
                score_force_persuasive    × 0.30
              + score_ancrage_auditoire   × 0.30
              + score_coherence_valeurs   × 0.25
              + (1 − score_risque_sophistique_rhethorique) × 0.15
            )
        Un profil élevé indique une argumentation rhétoriquement sophistiquée
        ET honnête — à distinguer d'une argumentation sophistiquée mais
        manipulatrice (force_persuasive haute, risque_sophistique haut).

    technique_dominante [str]
        La technique rhétorique de Perelman la plus utilisée dans la section.
        Valeurs possibles : voir TECHNIQUES_PERELMAN ci-dessous.

    type_auditoire [str]
        "universel" | "particulier_disciplinaire" | "particulier_ideologique" | "indetermine"

SORTIE JSON (trois niveaux, cohérente avec 07 et 09) :
    {
      "script"          : "10_map_perelman",
      "timestamp"       : "…",
      "modele_llm"      : "…",
      "score_global"    : 0.68,
      "profil_dominant" : "argument_autorite",
      "sections": [
        {
          "id"                                   : 1,
          "titre"                                : "…",
          "score_force_persuasive"               : 0.75,
          "score_ancrage_auditoire"              : 0.80,
          "score_coherence_valeurs"              : 0.70,
          "score_risque_sophistique_rhethorique" : 0.15,
          "score_profil_argumentatif"            : 0.74,
          "technique_dominante"                  : "argument_autorite",
          "type_auditoire"                       : "particulier_disciplinaire",
          "techniques_detectees": {
              "quasi_logique"     : ["incompatibilite", "comparaison"],
              "structure_reel"    : ["argument_autorite", "lien_causal"],
              "fonde_reel"        : ["illustration"],
              "dissociation"      : ["apparence_realite"]
          },
          "paires_dissociation"  : ["apparence / réalité"],
          "valeurs_mobilisees"   : ["objectivité", "représentativité"],
          "usages_sophistiques"  : ["auditoire_universel_abusif"],
          "analyse"              : "… texte brut LLM …",
          "passages_corpus"      : [{"source": "…", "page": 0, "extrait": "…"}]
        }
      ]
    }

UTILISATION :
    python 10_map_perelman_local.py
    python 10_map_perelman_local.py --map_file outputs/map_20240412_143022.json
    python 10_map_perelman_local.py --sections 2 5 8
    python 10_map_perelman_local.py --avec_argumentation outputs/argumentation_20240412.json

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
# (meilleure représentation du cadre Perelman dans les données d'entraînement).
from rag_config_local import LLM_MODEL, OLLAMA_URL

# Température légèrement supérieure à 09 : l'analyse rhétorique
# tolère une plus grande liberté interprétative.
TEMPERATURE = 0.2

# Nombre maximum de tokens en sortie par section.
# à augmenter si les sorties sont tronquées
MAX_TOKENS = 3500

# Nombre maximum de passages corpus inclus dans le prompt.
MAX_PASSAGES_PAR_SECTION = 6

# Répertoires.
MAP_DIR    = "outputs"
OUTPUT_DIR = "outputs"

# Pause entre deux appels Ollama.
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
# RÉFÉRENTIEL DES TECHNIQUES — Nouvelle Rhétorique
# =============================================================================

# Toutes les techniques reconnues par ce script, regroupées par famille.
TECHNIQUES_PERELMAN = {
    # A1 — Quasi-logiques
    "incompatibilite"  : "A1",
    "identite"         : "A1",
    "transitif"        : "A1",
    "reciprocite"      : "A1",
    "inclusion"        : "A1",
    "comparaison"      : "A1",
    "sacrifice"        : "A1",
    # A2 — Fondées sur la structure du réel
    "lien_causal"      : "A2",
    "argument_pragmatique": "A2",
    "argument_autorite": "A2",
    "illustration"     : "A2",
    "modele"           : "A2",
    "anti_modele"      : "A2",
    "analogie"         : "A2",
    # A3 — Fondent la structure du réel
    "exemple"          : "A3",
    "metaphore"        : "A3",
    # B — Dissociation
    "dissociation_apparence_realite" : "B",
    "dissociation_moyen_fin"         : "B",
    "dissociation_relatif_absolu"    : "B",
    "dissociation_individu_collectif": "B",
    "dissociation_lettre_esprit"     : "B",
    "dissociation_autre"             : "B",
}

# Familles pour le regroupement dans le JSON de sortie
FAMILLES = {
    "A1": "quasi_logique",
    "A2": "structure_reel",
    "A3": "fonde_reel",
    "B" : "dissociation",
}

# Usages sophistiques détectables
SOPHISTIQUES = [
    "auditoire_universel_abusif",  # prétendre à l'évidence universelle
    "metaphore_glissante",         # métaphore qui masque un glissement
    "fausse_dichotomie",           # dissociation qui simplifie abusivement
    "autorite_non_qualifiee",      # appel à l'autorité sans qualification
    "petition_principe",           # conclusion incluse dans les prémisses
]

# =============================================================================
# FONCTIONS — chargement
# =============================================================================

def trouver_dernier_map(map_dir: Path) -> Path:
    fichiers = sorted(map_dir.glob("map_*.json"), reverse=True)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier map_*.json dans {map_dir}. "
            "Lancez d'abord 06_map_enrich_local.py."
        )
    return fichiers[0]


def charger_map(chemin: Path) -> dict:
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    if "sections" not in data:
        raise ValueError(f"{chemin.name} ne contient pas de clé 'sections'.")
    return data


def charger_argumentation(chemin: Path) -> dict:
    """
    Charge optionnellement le fichier argumentation produit par 09.
    Retourne un dict {section_id: résultats_09} pour enrichissement croisé.
    """
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"]: s for s in data.get("sections", [])}


# =============================================================================
# FONCTIONS — construction du prompt
# =============================================================================

def formater_passages(passages: list[dict], max_passages: int) -> str:
    lignes = []
    for i, p in enumerate(passages[:max_passages], 1):
        source = p.get("source", "source inconnue")
        page   = p.get("page", "?")
        texte  = p.get("extrait", p.get("texte", "")).strip()
        if texte:
            lignes.append(f"[Passage {i} — {source}, p. {page}]\n{texte}")
    return "\n\n".join(lignes) if lignes else "(aucun passage corpus disponible)"


def construire_prompt(
    titre: str,
    texte: str,
    passages_str: str,
    scores_09: dict | None = None,
) -> str:
    """
    Construit le prompt d'analyse Perelman pour une section.

    Si scores_09 est fourni (résultats du script 09), ils sont inclus dans
    le prompt comme contexte pour l'analyse rhétorique, permettant au LLM
    de relier structure logique (Toulmin/Walton) et stratégie rhétorique (Perelman).
    """
    contexte_09 = ""
    if scores_09:
        r = scores_09.get("score_robustesse_globale", "N/A")
        s = scores_09.get("schema_walton_dominant", "N/A")
        w = scores_09.get("score_coherence_warrant", "N/A")
        contexte_09 = f"""
══════════════════════════════════════════
ANALYSE ARGUMENTATIVE (script 09 — Toulmin/Walton)
══════════════════════════════════════════
Robustesse logique : {r} | Schème Walton dominant : {s} | Cohérence du warrant : {w}
(Utilisez ces informations pour relier stratégie rhétorique et solidité logique.)
"""

    return f"""Tu es un expert en rhétorique académique, spécialisé dans la Nouvelle
Rhétorique de Perelman et Olbrechts-Tyteca. Analyse le passage suivant selon
ce cadre théorique, puis attribue les scores demandés.

══════════════════════════════════════════
SECTION DU MANUSCRIT : {titre}
══════════════════════════════════════════
{texte}
{contexte_09}
══════════════════════════════════════════
PASSAGES DU CORPUS (littérature secondaire)
══════════════════════════════════════════
{passages_str}

══════════════════════════════════════════
ANALYSE PERELMAN DEMANDÉE
══════════════════════════════════════════

━━━ 1. IDENTIFICATION DES TECHNIQUES ━━━
Identifie toutes les techniques argumentatives présentes dans le passage,
en les classant selon la Nouvelle Rhétorique :

A1 — QUASI-LOGIQUES (ressemblance formelle avec la logique) :
  Parmi : incompatibilite | identite | transitif | reciprocite | inclusion | comparaison | sacrifice
  Pour chaque technique : cite le passage exact qui l'illustre.

A2 — FONDÉES SUR LA STRUCTURE DU RÉEL :
  Parmi : lien_causal | argument_pragmatique | argument_autorite | illustration | modele | anti_modele | analogie
  Pour chaque technique : cite le passage exact.

A3 — QUI FONDENT LA STRUCTURE DU RÉEL :
  Parmi : exemple | metaphore
  Pour chaque technique : cite le passage exact.

B — DISSOCIATION CONCEPTUELLE (paires philosophiques) :
  Identifie les paires dissociées (apparence/réalité, moyen/fin, relatif/absolu,
  individu/collectif, lettre/esprit, ou autre).
  Pour chaque paire : quelle est la fonction argumentative de la dissociation ?

TECHNIQUE_DOMINANTE: <nom_technique>   (une seule, parmi les listes ci-dessus)

━━━ 2. ANALYSE DE L'AUDITOIRE ━━━
À quel type d'auditoire ce passage s'adresse-t-il ?
  TYPE_AUDITOIRE: universel | particulier_disciplinaire | particulier_ideologique | indetermine

Justifie : quels marqueurs textuels ou rhétoriques révèlent la construction de l'auditoire ?
L'auteur revendique-t-il abusivement l'auditoire universel (évidence, vérité générale)
pour ce qui relève d'un accord disciplinaire particulier ?

━━━ 3. VALEURS MOBILISÉES ━━━
Quelles valeurs l'argumentation présuppose-t-elle ou invoque-t-elle explicitement
(exemples : objectivité, progrès, causalité, représentativité, nation, modernité…) ?
VALEURS: <valeur1>, <valeur2>, …

Ces valeurs sont-elles cohérentes entre elles ? Y a-t-il tension ou contradiction ?

━━━ 4. USAGES SOPHISTIQUES ÉVENTUELS ━━━
Identifie les usages rhétoriques problématiques parmi :
  auditoire_universel_abusif | metaphore_glissante | fausse_dichotomie |
  autorite_non_qualifiee | petition_principe
USAGES_SOPHISTIQUES: <usage1>, <usage2>  (ou AUCUN)

━━━ 5. CONFRONTATION AVEC LE CORPUS ━━━
Les techniques d'autorité et d'illustration mobilisées dans le manuscrit
sont-elles confirmées ou infirmées par les passages corpus ?
L'auditoire construit par l'auteur correspond-il à celui du corpus ?

━━━ 6. SCORES (format strict) ━━━
  SCORE_FORCE_PERSUASIVE: X/10         (sophistication rhétorique pour l'auditoire disciplinaire)
  SCORE_ANCRAGE_AUDITOIRE: X/10        (adéquation aux conventions historiographiques)
  SCORE_COHERENCE_VALEURS: X/10        (cohérence du système de valeurs mobilisé)
  SCORE_RISQUE_SOPHISTIQUE: X/10       (0=aucun usage sophistique | 10=argumentation majoritairement sophistique)
  SCORE_PROFIL_ARGUMENTATIF: X/10      (synthèse : force×0.30 + ancrage×0.30 + valeurs×0.25 + (10−risque)×0.15, divise par 10)

━━━ 7. SYNTHÈSE RHÉTORIQUE ━━━
En 3–4 phrases : comment ce passage construit-il sa persuasion ?
Quelles techniques dominent ? Y a-t-il inadéquation entre ambition rhétorique
et auditoire réel ? Quelle révision permettrait de renforcer la force persuasive
sans glisser vers la sophistique ?"""


# =============================================================================
# FONCTIONS — appel LLM
# =============================================================================

def appel_ollama(prompt: str) -> str:
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
        print("  ❌ Ollama inaccessible. Vérifiez : ollama serve", file=sys.stderr)
        return ""
    except requests.exceptions.Timeout:
        print("  ❌ Timeout Ollama (>120 s). Section ignorée.", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  ❌ Erreur Ollama : {e}", file=sys.stderr)
        return ""


# =============================================================================
# FONCTIONS — extraction
# =============================================================================

PATTERNS_SCORES = {
    "score_force_persuasive"               : r"SCORE_FORCE_PERSUASIVE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_ancrage_auditoire"              : r"SCORE_ANCRAGE_AUDITOIRE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_coherence_valeurs"              : r"SCORE_COHERENCE_VALEURS\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_risque_sophistique_rhethorique" : r"SCORE_RISQUE_SOPHISTIQUE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_profil_argumentatif"            : r"SCORE_PROFIL_ARGUMENTATIF\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
}

NOMS_COURTS_SCORES = {
    "score_force_persuasive"               : ["force_persuasive", "force persuasive", "persuasive"],
    "score_ancrage_auditoire"              : ["ancrage_auditoire", "ancrage auditoire", "ancrage"],
    "score_coherence_valeurs"              : ["cohérence_valeurs", "coherence_valeurs", "cohérence valeurs"],
    "score_risque_sophistique_rhethorique" : ["sophistique", "sophistique_rhétorique", "risque_sophistique"],
    "score_profil_argumentatif"            : ["profil_argumentatif", "profil argumentatif", "profil"],
}

VALEUR_NEUTRE = 0.5


def _normaliser_score(valeur: float, avec_denominateur: bool) -> float:
    """Normalise un score brut vers [0.0, 1.0]."""
    if avec_denominateur or valeur > 1.0:
        valeur = valeur / 10.0
    return round(max(0.0, min(1.0, valeur)), 3)


def extraire_scores(texte_llm: str) -> dict:
    """
    Extrait les cinq scores Perelman depuis la réponse LLM brute.

    Quatre niveaux de robustesse :
    Niveau 1 — Format strict : SCORE_X: N/10 (virgule ou point acceptés)
    Niveau 2 — Espaces autour du / (couvert par niveau 1)
    Niveau 3 — Nom court sans préfixe SCORE_
    Niveau 4 — Tableau Markdown | label | valeur |
    Repli    — VALEUR_NEUTRE (0.5) si aucun niveau ne trouve le score.
    """
    scores = {}

    for nom, pattern in PATTERNS_SCORES.items():
        # Niveau 1 + 2
        m = re.search(pattern, texte_llm, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(",", "."))
            scores[nom] = _normaliser_score(v, avec_denominateur=True)
            continue

        # Niveau 3 — nom court
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


def calculer_profil(scores: dict) -> float:
    """
    Recalcule score_profil_argumentatif selon la pondération documentée.
    Utilisé pour vérifier (et corriger) la valeur fournie par le LLM.
    """
    profil = (
        scores.get("score_force_persuasive", 0.5)               * 0.30
        + scores.get("score_ancrage_auditoire", 0.5)            * 0.30
        + scores.get("score_coherence_valeurs", 0.5)            * 0.25
        + (1 - scores.get("score_risque_sophistique_rhethorique", 0.5)) * 0.15
    )
    return round(profil, 3)


def extraire_technique_dominante(texte_llm: str) -> str:
    m = re.search(r"TECHNIQUE_DOMINANTE\s*:\s*(\w+)", texte_llm, re.IGNORECASE)
    if m:
        tech = m.group(1).lower()
        return tech if tech in TECHNIQUES_PERELMAN else "indeterminee"
    return "indeterminee"


def extraire_type_auditoire(texte_llm: str) -> str:
    m = re.search(r"TYPE_AUDITOIRE\s*:\s*(\S+)", texte_llm, re.IGNORECASE)
    if m:
        t = m.group(1).lower().strip(".,;")
        valides = {"universel", "particulier_disciplinaire", "particulier_ideologique", "indetermine"}
        return t if t in valides else "indetermine"
    return "indetermine"


def extraire_valeurs(texte_llm: str) -> list[str]:
    m = re.search(r"VALEURS\s*:\s*(.+)", texte_llm, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        return [v.strip() for v in re.split(r"[,;]", raw) if v.strip()]
    return []


def extraire_usages_sophistiques(texte_llm: str) -> list[str]:
    m = re.search(r"USAGES_SOPHISTIQUES\s*:\s*(.+)", texte_llm, re.IGNORECASE)
    if m:
        raw = m.group(1).strip()
        if raw.upper() == "AUCUN":
            return []
        candidats = [s.strip().lower() for s in re.split(r"[,;]", raw)]
        return [s for s in candidats if s in SOPHISTIQUES]
    return []


def extraire_techniques_par_famille(texte_llm: str) -> dict:
    """
    Tente d'identifier les techniques mentionnées dans le texte LLM
    et les regroupe par famille (A1, A2, A3, B).
    Méthode simple : présence du nom de la technique dans le texte.
    """
    par_famille = {famille: [] for famille in FAMILLES.values()}
    for technique, famille_code in TECHNIQUES_PERELMAN.items():
        nom_famille = FAMILLES[famille_code]
        # Recherche du nom de la technique dans le texte (insensible à la casse)
        if re.search(rf"\b{re.escape(technique)}\b", texte_llm, re.IGNORECASE):
            if technique not in par_famille[nom_famille]:
                par_famille[nom_famille].append(technique)
    return par_famille


def extraire_paires_dissociation(texte_llm: str) -> list[str]:
    """
    Extrait les paires de dissociation mentionnées dans la réponse LLM.
    Cherche les patterns "X / Y" ou "X/Y" communs à la Nouvelle Rhétorique.
    """
    paires_connues = [
        "apparence / réalité", "moyen / fin", "relatif / absolu",
        "individu / collectif", "lettre / esprit",
        "subjectif / objectif", "acte / personne",
        "théorie / pratique", "intérêt personnel / intérêt général",
    ]
    trouvees = []
    for paire in paires_connues:
        # Cherche la paire avec ou sans espaces autour du /
        pattern = paire.replace(" / ", r"\s*/\s*")
        if re.search(pattern, texte_llm, re.IGNORECASE):
            trouvees.append(paire)
    return trouvees


# =============================================================================
# FONCTION — traitement d'une section
# =============================================================================

def analyser_section(section: dict, scores_09: dict | None = None) -> dict | None:
    """
    Orchestre l'analyse Perelman d'une section.

    Args:
        section   : dict de la section issu du fichier map.
        scores_09 : optionnel, dict des scores produits par 09 pour la même section.
    """
    titre    = section.get("titre", f"Section {section.get('id', '?')}")
    texte    = section.get("texte", "")
    passages = section.get("passages_corpus", [])

    if not texte.strip():
        print("  ⚠ Section sans texte — ignorée.", file=sys.stderr)
        return None

    passages_str  = formater_passages(passages, MAX_PASSAGES_PAR_SECTION)
    prompt        = construire_prompt(titre, texte, passages_str, scores_09)
    reponse_brute = appel_ollama(prompt)

    if not reponse_brute:
        return None

    # Extraction
    scores               = extraire_scores(reponse_brute)
    technique_dominante  = extraire_technique_dominante(reponse_brute)
    type_auditoire       = extraire_type_auditoire(reponse_brute)
    valeurs              = extraire_valeurs(reponse_brute)
    usages_sophistiques  = extraire_usages_sophistiques(reponse_brute)
    techniques_familles  = extraire_techniques_par_famille(reponse_brute)
    paires_dissociation  = extraire_paires_dissociation(reponse_brute)

    # Vérification du profil argumentatif
    profil_verif = calculer_profil(scores)
    ecart = abs(scores.get("score_profil_argumentatif", 0.5) - profil_verif)
    if ecart > 0.15:
        print(
            f"  ℹ Profil LLM corrigé : {scores['score_profil_argumentatif']:.3f}"
            f" → {profil_verif:.3f} (écart {ecart:.3f})",
            file=sys.stderr,
        )
        scores["score_profil_argumentatif"] = profil_verif

    return {
        "id"                                   : section.get("id"),
        "titre"                                : titre,
        **scores,
        "technique_dominante"                  : technique_dominante,
        "type_auditoire"                       : type_auditoire,
        "techniques_detectees"                 : techniques_familles,
        "paires_dissociation"                  : paires_dissociation,
        "valeurs_mobilisees"                   : valeurs,
        "usages_sophistiques"                  : usages_sophistiques,
        "analyse"                              : reponse_brute,
        "passages_corpus": [
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
    if not sections_analysees:
        return 0.0, "indeterminee"

    profils = [s["score_profil_argumentatif"] for s in sections_analysees]
    score_global = round(sum(profils) / len(profils), 3)

    comptage = {}
    for s in sections_analysees:
        tech = s.get("technique_dominante", "indeterminee")
        comptage[tech] = comptage.get(tech, 0) + 1
    technique_dominante = max(comptage, key=comptage.get)

    return score_global, technique_dominante


# =============================================================================
# SAUVEGARDE
# =============================================================================

def sauvegarder(
    sections_analysees: list[dict],
    score_global: float,
    profil_dominant: str,
    output_dir: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin    = output_dir / f"perelman_{timestamp}.json"

    sortie = {
        "script"          : "10_map_perelman",
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
    print("\n" + "═" * 76)
    print(f"  BILAN — 10_map_perelman | Score profil global : {score_global:.3f}")
    print("═" * 76)
    print(
        f"  {'§':>3}  {'Titre':<28}  {'Force':>6}  {'Ancrag':>6}  "
        f"{'Valeur':>6}  {'Risq.':>6}  {'Profil':>6}  Technique"
    )
    print("─" * 76)
    for s in sections_analysees:
        print(
            f"  {s['id']:>3}  {s['titre'][:28]:<28}  "
            f"{s['score_force_persuasive']:.2f}    "
            f"{s['score_ancrage_auditoire']:.2f}    "
            f"{s['score_coherence_valeurs']:.2f}    "
            f"{s['score_risque_sophistique_rhethorique']:.2f}    "
            f"{s['score_profil_argumentatif']:.2f}  "
            f"{s.get('technique_dominante', '?')}"
        )
        if s.get("usages_sophistiques"):
            print(f"       ⚠ Sophistiques : {', '.join(s['usages_sophistiques'])}")
        if s.get("paires_dissociation"):
            print(f"       ↔ Dissociations : {', '.join(s['paires_dissociation'])}")
    print("═" * 76 + "\n")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="10_map_perelman — Analyse Nouvelle Rhétorique du manuscrit."
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default=None,
        help="Chemin vers le fichier map JSON produit par 06_map_enrich_local.py.",
    )
    parser.add_argument(
        "--sections",
        type=int,
        nargs="+",
        default=None,
        metavar="ID",
        help="Identifiants des sections à analyser. Si absent : toutes les sections.",
    )
    parser.add_argument(
        "--avec_argumentation",
        type=str,
        default=None,
        metavar="CHEMIN",
        help="Chemin vers le fichier argumentation_{timestamp}.json produit par 09. "
             "Si fourni, les scores Toulmin/Walton sont inclus dans le prompt Perelman.",
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

    # Chargement optionnel des résultats 09
    scores_09_par_section = {}
    if args.avec_argumentation:
        chemin_09 = Path(args.avec_argumentation)
        if chemin_09.exists():
            scores_09_par_section = charger_argumentation(chemin_09)
            print(f"  Résultats 09 chargés : {chemin_09.name} ({len(scores_09_par_section)} sections)")
        else:
            print(f"  ⚠ Fichier argumentation introuvable : {chemin_09}", file=sys.stderr)

    # Filtre optionnel
    if args.sections:
        sections = [s for s in sections if s.get("id") in args.sections]
        if not sections:
            print(f"❌ Aucune section avec les identifiants {args.sections}.")
            sys.exit(1)

    print(f"  {len(sections)} section(s) à analyser.")
    print(f"  Modèle LLM : {LLM_MODEL} | T° : {TEMPERATURE}\n")

    # Boucle d'analyse
    sections_analysees = []
    for sec in sections:
        sid = sec.get("id")
        print(f"§ {sid} — {sec.get('titre', '?')}")
        scores_09 = scores_09_par_section.get(sid)
        resultat  = analyser_section(sec, scores_09)
        if resultat:
            p = resultat["score_profil_argumentatif"]
            t = resultat.get("technique_dominante", "?")
            print(f"  Profil : {p:.3f} | Technique dominante : {t}")
            sections_analysees.append(resultat)
        else:
            print("  ⚠ Section ignorée (réponse vide ou texte manquant).")
        time.sleep(PAUSE_INTER_SECTION)
        print()

    if not sections_analysees:
        print("❌ Aucune section analysée. Vérifiez le fichier map et Ollama.")
        sys.exit(1)

    score_global, profil_dominant = calculer_score_global(sections_analysees)
    chemin_sortie = sauvegarder(sections_analysees, score_global, profil_dominant, output_dir)

    afficher_bilan(sections_analysees, score_global)
    print(f"✅ Analyse sauvegardée : {chemin_sortie.name}")
    print(f"   Score profil global : {score_global:.3f}")
    print(f"   Technique dominante (manuscrit) : {profil_dominant}\n")


if __name__ == "__main__":
    main()
