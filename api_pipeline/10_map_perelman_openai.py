"""
10_map_perelman_openai.py
==========================
Étape 10 du pipeline RAG — Analyse rhétorique selon la Nouvelle Rhétorique.
VERSION OPENAI (API distante, modèles gpt-4.1 / gpt-4.1-mini).

Rôle : Pour chaque section du manuscrit (issue du fichier map produit par
06_map_enrich.py), analyse les techniques argumentatives selon le cadre de
la Nouvelle Rhétorique de Chaïm Perelman et Lucie Olbrechts-Tyteca.

RÉFÉRENCE THÉORIQUE PRINCIPALE :
    Chaïm Perelman & Lucie Olbrechts-Tyteca,
    Traité de l'argumentation — La Nouvelle Rhétorique.
    Bruxelles : Éditions de l'Université de Bruxelles, 1958 (rééd. 1988).

POSITION DANS LE PIPELINE :
    06_map_enrich.py           → map_{timestamp}.json
    09_map_argumentation_openai.py → argumentation_{timestamp}.json
    10_map_perelman_openai.py  → perelman_{timestamp}.json  (ce script)

DIFFÉRENCES AVEC LA VERSION LOCALE (10_map_perelman.py) :
    - Utilise LLMClient depuis 00_config.py
    - Modèle par défaut : gpt-4.1-mini
    - Nécessite OPENAI_API_KEY dans .env
    - Coût indicatif : 0.02–0.10 $ par analyse avec gpt-4.1-mini
    - System_prompt stable pour mise en cache de préfixe OpenAI

CADRE THÉORIQUE — NOUVELLE RHÉTORIQUE (résumé) :

    Perelman distingue deux grands types de techniques argumentatives :

    A. TECHNIQUES D'ASSOCIATION — rapprocher des éléments pour les rendre solidaires

       A1. Quasi-logiques : ressemblance avec la logique formelle
           incompatibilite | identite | transitif | reciprocite |
           inclusion | comparaison | sacrifice

       A2. Fondées sur la structure du réel : liaisons reconnues dans le réel
           lien_causal | argument_pragmatique | argument_autorite |
           illustration | modele | anti_modele | analogie

       A3. Qui fondent la structure du réel : créer de nouvelles liaisons
           exemple | metaphore

    B. TECHNIQUES DE DISSOCIATION — scinder un concept en deux termes hiérarchisés
       Paires philosophiques (terme I dévalué, terme II valorisé) :
           apparence/réalité | moyen/fin | relatif/absolu |
           individu/collectif | lettre/esprit | subjectif/objectif |
           acte/personne | théorie/pratique

    C. LA NOTION D'AUDITOIRE (centrale chez Perelman)
       Auditoire universel : l'argument prétend s'adresser à tout être raisonnable.
       Auditoire particulier : l'argument cible un groupe aux valeurs communes.
       En histoire académique : auditoire particulier disciplinaire
       (communauté des historiens, avec ses conventions épistémiques propres).

SCORES PRODUITS (tous entre 0.0 et 1.0) :

    score_force_persuasive [0–1]
        Efficacité rhétorique du passage pour son auditoire académique.
        0.0 = texte purement assertif, sans dispositif rhétorique.
        1.0 = argumentation sophistiquée, multiplicité de techniques articulées.
        ⚠ Limite : mesure la sophistication rhétorique, pas la vérité de l'argument
        (voir 09 pour la robustesse logique).

    score_ancrage_auditoire [0–1]
        Adéquation aux conventions de l'auditoire disciplinaire des historiens :
        débats historiographiques, normes de citation, critique des sources.
        0.0 = ignorance des codes disciplinaires.
        1.0 = maîtrise parfaite des conventions rhétoriques de la discipline.
        ⚠ Limite : dépend de la richesse du corpus ; sous-estimé si le corpus
        est lacunaire dans le sous-domaine traité.

    score_coherence_valeurs [0–1]
        Cohérence interne du système de valeurs mobilisé par l'auteur :
        objectivité, progrès, représentativité, causalité, etc.
        0.0 = valeurs contradictoires ou opportunistes.
        1.0 = système de valeurs cohérent et explicite.
        ⚠ Limite : détecte les incohérences formelles, pas l'épistémoligie 
        effectivement mise en oeuvre.
        Le script détecte des éléments des stratégies rhétoriques déployées
        pas les pratiques effectives.

    score_risque_sophistique_rhethorique [0–1]
        Usages rhétoriques problématiques au sens de Perelman :
        auditoire universel revendiqué abusivement, métaphore glissante,
        fausse dichotomie, autorité non qualifiée, pétition de principe.
        0.0 = aucun usage sophistique.
        1.0 = argumentation reposant principalement sur des effets non fondés.
        ⚠ Limite : frontière contextuelle entre rhétorique efficace et sophisme.

    score_profil_argumentatif [0–1]
        Synthèse pondérée :
            force×0.30 + ancrage×0.30 + valeurs×0.25 + (1−risque)×0.15
        Distingue l'argumentation sophistiquée ET honnête de l'argumentation
        sophistiquée mais manipulatrice (force haute + risque haut).

    technique_dominante [str]
        Technique Perelman la plus représentée dans la section.

    type_auditoire [str]
        "universel" | "particulier_disciplinaire" |
        "particulier_ideologique" | "indetermine"

SORTIE JSON (trois niveaux, cohérente avec 07 et 09) :
    {
      "script"          : "10_map_perelman_openai",
      "timestamp"       : "…",
      "modele_llm"      : "gpt-4.1-mini",
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
          "techniques_detectees"  : {"quasi_logique":[],"structure_reel":[],"fonde_reel":[],"dissociation":[]},
          "paires_dissociation"   : [],
          "valeurs_mobilisees"    : ["objectivité","représentativité"],
          "usages_sophistiques"   : [],
          "analyse"               : "…texte brut LLM…",
          "passages_corpus"       : [{"source":"…","page":0,"extrait":"…"}]
        }
      ]
    }

UTILISATION :
    python 10_map_perelman_openai.py
    python 10_map_perelman_openai.py --map_file outputs/map_20240412.json
    python 10_map_perelman_openai.py --sections 2 5 8
    python 10_map_perelman_openai.py --avec_argumentation outputs/argumentation_20240412.json

Environnement conda : rag_historien
Prérequis : fichier .env avec OPENAI_API_KEY=sk-...
"""

# =============================================================================
# PARAMÈTRES — modifier ici sans toucher au reste du script
# =============================================================================

# Modèle OpenAI. "gpt-4.1-mini" pour l'usage courant, "gpt-4.1" pour le final.
OPENAI_LLM_MODEL = "gpt-4.1-mini"

# Température. Légèrement supérieure à 09 : l'analyse rhétorique tolère
# une interprétation plus nuancée que l'analyse logique.
TEMPERATURE = 0.2

# Tokens max en sortie par section.
MAX_TOKENS = 2500

# Passages corpus max dans le prompt (impacte le coût API).
MAX_PASSAGES_PAR_SECTION = 8

# Répertoires.
MAP_DIR    = "resultats"
OUTPUT_DIR = "resultats"

# Pause entre deux appels API.
PAUSE_INTER_SECTION = 0.5

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

from dotenv import load_dotenv
from config_00 import LLMClient

load_dotenv()

# =============================================================================
# RÉFÉRENTIEL PERELMAN
# =============================================================================

TECHNIQUES_PERELMAN = {
    # A1
    "incompatibilite": "A1", "identite": "A1", "transitif": "A1",
    "reciprocite": "A1", "inclusion": "A1", "comparaison": "A1", "sacrifice": "A1",
    # A2
    "lien_causal": "A2", "argument_pragmatique": "A2", "argument_autorite": "A2",
    "illustration": "A2", "modele": "A2", "anti_modele": "A2", "analogie": "A2",
    # A3
    "exemple": "A3", "metaphore": "A3",
    # B
    "dissociation_apparence_realite": "B", "dissociation_moyen_fin": "B",
    "dissociation_relatif_absolu": "B", "dissociation_individu_collectif": "B",
    "dissociation_lettre_esprit": "B", "dissociation_autre": "B",
}

FAMILLES = {"A1": "quasi_logique", "A2": "structure_reel", "A3": "fonde_reel", "B": "dissociation"}

SOPHISTIQUES = [
    "auditoire_universel_abusif", "metaphore_glissante", "fausse_dichotomie",
    "autorite_non_qualifiee", "petition_principe",
]

# Prompt système stable (mise en cache de préfixe OpenAI).
SYSTEM_PROMPT = (
    "Tu es un expert en rhétorique académique, spécialisé dans la Nouvelle Rhétorique "
    "de Perelman et Olbrechts-Tyteca (Traité de l'argumentation, 1958). Tu analyses "
    "les techniques argumentatives des textes historiques avec précision, en distinguant "
    "techniques d'association, techniques de dissociation et construction de l'auditoire. "
    "Tu réponds dans la langue du texte analysé ou en français si le texte est multilingue."
)

# =============================================================================
# FONCTIONS — chargement
# =============================================================================

def trouver_dernier_map(map_dir: Path) -> Path:
    fichiers = sorted(map_dir.glob("enrich*.json"), reverse=True)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier enrich*.json dans {map_dir}.\n"
            "Lancez d'abord 06_map_enrich.py."
        )
    return fichiers[0]


def charger_map(chemin: Path) -> dict:
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    if "paragraphes" not in data:
        raise ValueError(f"{chemin.name} ne contient pas de clé 'paragraphes'.")
    return data


def charger_argumentation(chemin: Path) -> dict:
    """
    Charge optionnellement les résultats du script 09.
    Retourne un dict {paragraphe_id: résultats_09} pour enrichissement croisé.
    Le croisement permet au LLM de relier solidité logique et stratégie rhétorique.
    """
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    return {s["id"]: s for s in data.get("paragraphes", [])}


# =============================================================================
# FONCTIONS — construction du prompt
# =============================================================================

def formater_passages(passages: list[dict], max_passages: int) -> str:
    lignes = []
    for i, p in enumerate(passages[:max_passages], 1):
        source = p.get("ref_courte", p.get("source", "source inconnue"))
        page   = p.get("page", "?")
        texte  = p.get("extrait", p.get("texte", "")).strip()
        if texte:
            lignes.append(f"[Passage {i} — {source}, p. {page}]\n{texte}")
    return "\n\n".join(lignes) if lignes else "(aucun passage corpus disponible)"


def construire_user_message(
    titre: str,
    texte: str,
    passages_str: str,
    scores_09: dict | None = None,
) -> str:
    """
    Message utilisateur pour l'analyse Perelman.
    Si scores_09 est fourni, les résultats Toulmin/Walton sont injectés
    dans le prompt pour permettre un croisement des deux analyses.
    """
    contexte_09 = ""
    if scores_09:
        r = scores_09.get("score_robustesse_globale", "N/A")
        s = scores_09.get("schema_walton_dominant", "N/A")
        w = scores_09.get("score_coherence_warrant", "N/A")
        contexte_09 = (
            f"\n══════════════════════════════════════════\n"
            f"CONTEXTE — ANALYSE ARGUMENTATIVE (script 09)\n"
            f"══════════════════════════════════════════\n"
            f"Robustesse logique (Toulmin/Walton) : {r} | "
            f"Schème dominant : {s} | Cohérence warrant : {w}\n"
            f"(Reliez la stratégie rhétorique à cette solidité logique.)\n"
        )

    return f"""Analyse le passage suivant selon la Nouvelle Rhétorique de Perelman,
puis attribue les scores demandés.

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

━━━ 1. TECHNIQUES D'ASSOCIATION ━━━
A1 — QUASI-LOGIQUES :
  Parmi : incompatibilite | identite | transitif | reciprocite | inclusion | comparaison | sacrifice
  Pour chaque technique détectée : cite le passage exact.

A2 — FONDÉES SUR LA STRUCTURE DU RÉEL :
  Parmi : lien_causal | argument_pragmatique | argument_autorite | illustration | modele | anti_modele | analogie
  Pour chaque technique détectée : cite le passage exact.

A3 — QUI FONDENT LA STRUCTURE DU RÉEL :
  Parmi : exemple | metaphore
  Pour chaque technique détectée : cite le passage exact.

TECHNIQUE_DOMINANTE: <nom_technique>   (un seul nom parmi les listes ci-dessus)

━━━ 2. TECHNIQUES DE DISSOCIATION ━━━
B — PAIRES PHILOSOPHIQUES :
  Identifie les dissociations conceptuelles (apparence/réalité, moyen/fin,
  relatif/absolu, individu/collectif, lettre/esprit, ou autre).
  Pour chaque paire : quelle est sa fonction argumentative dans ce passage ?

━━━ 3. CONSTRUCTION DE L'AUDITOIRE ━━━
TYPE_AUDITOIRE: universel | particulier_disciplinaire | particulier_ideologique | indetermine
Justifie : quels marqueurs textuels révèlent la construction de l'auditoire ?
L'auteur revendique-t-il abusivement l'auditoire universel ?

━━━ 4. VALEURS MOBILISÉES ━━━
VALEURS: <valeur1>, <valeur2>, …
Ces valeurs sont-elles cohérentes ? Y a-t-il tension ou contradiction ?

━━━ 5. USAGES SOPHISTIQUES ━━━
Parmi : auditoire_universel_abusif | metaphore_glissante | fausse_dichotomie |
        autorite_non_qualifiee | petition_principe
USAGES_SOPHISTIQUES: <usage1>, <usage2>  (ou AUCUN)

━━━ 6. CONFRONTATION CORPUS ━━━
Les techniques d'autorité et d'illustration du manuscrit sont-elles confirmées
ou infirmées par les passages corpus ?

━━━ 7. SCORES (format strict) ━━━
  SCORE_FORCE_PERSUASIVE: X/10
  SCORE_ANCRAGE_AUDITOIRE: X/10
  SCORE_COHERENCE_VALEURS: X/10
  SCORE_RISQUE_SOPHISTIQUE: X/10
  SCORE_PROFIL_ARGUMENTATIF: X/10  ← force×0.30 + ancrage×0.30 + valeurs×0.25 + (10−risque)×0.15, divisé par 10

━━━ 8. SYNTHÈSE RHÉTORIQUE ━━━
En 3–4 phrases : comment ce passage construit-il sa persuasion ?
Quelles techniques dominent ? Inadéquation auditoire/ambition rhétorique ?
Quelle révision renforcerait la force persuasive sans glisser vers la sophistique ?"""


# =============================================================================
# FONCTIONS — extraction
# =============================================================================

PATTERNS_SCORES = {
    "score_force_persuasive"               : r"SCORE_FORCE_PERSUASIVE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_ancrage_auditoire"              : r"SCORE_ANCRAGE_AUDITOIRE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_coherence_valeurs"              : r"SCORE_COHERENCE_VALEURS\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_risque_sophistique_rhethorique" : r"SCORE_RISQUE_SOPHISTIQUE\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
    "score_profil_argumentatif"            : r"SCORE_PROFIL_ARGUMENTATIF\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10",
}

VALEUR_NEUTRE = 0.5


def extraire_scores(texte_llm: str) -> dict:
    scores = {}
    for nom, pattern in PATTERNS_SCORES.items():
        m = re.search(pattern, texte_llm, re.IGNORECASE)
        if m:
            v = max(0.0, min(10.0, float(m.group(1))))
            scores[nom] = round(v / 10.0, 3)
        else:
            print(f"  ⚠ Score non trouvé : {nom} → {VALEUR_NEUTRE}", file=sys.stderr)
            scores[nom] = VALEUR_NEUTRE
    return scores


def calculer_profil(scores: dict) -> float:
    return round(
        scores.get("score_force_persuasive", 0.5) * 0.30
        + scores.get("score_ancrage_auditoire", 0.5) * 0.30
        + scores.get("score_coherence_valeurs", 0.5) * 0.25
        + (1 - scores.get("score_risque_sophistique_rhethorique", 0.5)) * 0.15,
        3,
    )


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
        valides = {"universel","particulier_disciplinaire","particulier_ideologique","indetermine"}
        return t if t in valides else "indetermine"
    return "indetermine"


def extraire_valeurs(texte_llm: str) -> list[str]:
    m = re.search(r"VALEURS\s*:\s*(.+)", texte_llm, re.IGNORECASE)
    if m:
        return [v.strip() for v in re.split(r"[,;]", m.group(1)) if v.strip()]
    return []


def extraire_usages_sophistiques(texte_llm: str) -> list[str]:
    m = re.search(r"USAGES_SOPHISTIQUES\s*:\s*(.+)", texte_llm, re.IGNORECASE)
    if m and m.group(1).strip().upper() != "AUCUN":
        return [s.strip().lower() for s in re.split(r"[,;]", m.group(1))
                if s.strip().lower() in SOPHISTIQUES]
    return []


def extraire_techniques_par_famille(texte_llm: str) -> dict:
    par_famille = {f: [] for f in FAMILLES.values()}
    for technique, code in TECHNIQUES_PERELMAN.items():
        nom = FAMILLES[code]
        if re.search(rf"\b{re.escape(technique)}\b", texte_llm, re.IGNORECASE):
            if technique not in par_famille[nom]:
                par_famille[nom].append(technique)
    return par_famille


def extraire_paires_dissociation(texte_llm: str) -> list[str]:
    paires = [
        "apparence / réalité", "moyen / fin", "relatif / absolu",
        "individu / collectif", "lettre / esprit", "subjectif / objectif",
        "acte / personne", "théorie / pratique", "intérêt personnel / intérêt général",
    ]
    return [p for p in paires
            if re.search(p.replace(" / ", r"\s*/\s*"), texte_llm, re.IGNORECASE)]


# =============================================================================
# GÉNÉRATION DU RAPPORT MARKDOWN
# =============================================================================

def _barre(valeur: float, largeur: int = 20) -> str:
    """Barre ASCII proportionnelle. Ex : [████████░░░░░░░░░░░░] 0.42"""
    rempli = round(valeur * largeur)
    return f"[{'█' * rempli}{'░' * (largeur - rempli)}] {valeur:.2f}"


def generer_rapport_md(
    paragraphes: list[dict],
    score_global: float,
    profil: str,
    timestamp: str,
    modele: str,
    output_dir: Path,
) -> Path:
    """
    Génère perelman_{timestamp}.md — rapport lisible par section.

    Structure par section :
      - Extrait du texte analysé (si disponible dans passages_corpus)
      - Techniques Perelman par famille (A1/A2/A3/B) avec citations LLM
      - Paires de dissociation identifiées
      - Valeurs mobilisées
      - Type d'auditoire
      - Usages sophistiques détectés
      - Scores numériques avec barres ASCII
      - Synthèse LLM (3–4 phrases)

    Produit en parallèle du JSON dans OUTPUT_DIR.
    """

    def _extraire_synthese(analyse_brute: str) -> str:
        m = re.search(
            r"(?:synthèse rhétorique|SYNTHÈSE RHÉTORIQUE|synthèse|SYNTHÈSE)[^\n]*\n(.+?)(?:\n\s*━|$)",
            analyse_brute,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        return analyse_brute.strip()[-600:] if analyse_brute else "(non disponible)"

    NOMS_FAMILLES = {
        "quasi_logique" : "A1 — Quasi-logiques",
        "structure_reel": "A2 — Fondées sur la structure du réel",
        "fonde_reel"    : "A3 — Qui fondent la structure du réel",
        "dissociation"  : "B  — Dissociation",
    }

    lignes = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lignes += [
        "# Rapport d'analyse rhétorique — Nouvelle Rhétorique (Perelman)",
        "",
        f"**Script** : `10_map_perelman_openai`  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {modele}  ",
        f"**Température** : {TEMPERATURE}  ",
        f"**Sections analysées** : {len(paragraphes)}  ",
        "",
        "---",
        "",
        "## Bilan global",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| Score de profil argumentatif global | **{score_global:.3f}** |",
        f"| Technique dominante | `{profil}` |",
        "",
    ]

    # Tableau récapitulatif
    lignes += [
        "| § | Titre | Force | Ancrage | Valeurs | Risque | Profil | Technique |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in paragraphes:
        lignes.append(
            f"| {p['id']} "
            f"| {p.get('titre','')[:35]} "
            f"| {p['score_force_persuasive']:.2f} "
            f"| {p['score_ancrage_auditoire']:.2f} "
            f"| {p['score_coherence_valeurs']:.2f} "
            f"| {p['score_risque_sophistique_rhethorique']:.2f} "
            f"| **{p['score_profil_argumentatif']:.2f}** "
            f"| `{p.get('technique_dominante','?')}` |"
        )
    lignes += ["", "---", ""]

    # ── Sections détaillées ───────────────────────────────────────────────────
    lignes.append("## Analyse par section\n")

    for p in paragraphes:
        idx   = p.get("id", "?")
        titre = p.get("titre", f"Paragraphe {idx}")
        analyse_brute = p.get("analyse", "")

        # Extrait du corpus (premier passage disponible)
        extrait_texte = ""
        for pas in p.get("passages_corpus", []):
            if pas.get("extrait"):
                extrait_texte = pas["extrait"][:200]
                break

        lignes += [f"### § {idx} — {titre}", ""]

        if extrait_texte:
            lignes += [
                f"> *{extrait_texte}{'…' if len(extrait_texte) == 200 else ''}*",
                "",
            ]

        # ── Techniques par famille ────────────────────────────────────────────
        lignes += ["#### Techniques Perelman détectées", ""]
        techniques = p.get("techniques_detectees", {})
        for cle_famille, label_famille in NOMS_FAMILLES.items():
            items = techniques.get(cle_famille, [])
            if items:
                lignes.append(f"**{label_famille}**")
                for tech in items:
                    lignes.append(f"- `{tech}`")
            else:
                lignes.append(f"**{label_famille}** : *(aucune détectée)*")
        lignes.append("")

        # ── Paires de dissociation ────────────────────────────────────────────
        paires = p.get("paires_dissociation", [])
        lignes += ["#### Paires de dissociation", ""]
        if paires:
            for paire in paires:
                lignes.append(f"- {paire}")
        else:
            lignes.append("*(aucune paire identifiée)*")
        lignes.append("")

        # ── Valeurs mobilisées ────────────────────────────────────────────────
        valeurs = p.get("valeurs_mobilisees", [])
        lignes += ["#### Valeurs mobilisées", ""]
        if valeurs:
            lignes.append(", ".join(f"`{v}`" for v in valeurs))
        else:
            lignes.append("*(aucune valeur identifiée)*")
        lignes.append("")

        # ── Type d'auditoire ──────────────────────────────────────────────────
        auditoire = p.get("type_auditoire", "indetermine")
        lignes += ["#### Type d'auditoire", ""]
        label_auditoire = {
            "universel"                 : "Universel — prétend s'adresser à tout être raisonnable",
            "particulier_disciplinaire" : "Particulier disciplinaire — communauté des historiens",
            "particulier_ideologique"   : "Particulier idéologique — groupe aux valeurs communes",
            "indetermine"               : "Indéterminé",
        }.get(auditoire, auditoire)
        lignes.append(f"`{auditoire}` — {label_auditoire}")

        # Usages sophistiques
        usages = p.get("usages_sophistiques", [])
        if usages:
            lignes += ["", f"⚠ **Usages sophistiques** : {', '.join(f'`{u}`' for u in usages)}"]
        lignes.append("")

        # ── Scores ────────────────────────────────────────────────────────────
        lignes += ["#### Scores", ""]
        scores_affich = [
            ("Force persuasive",        p.get("score_force_persuasive", 0)),
            ("Ancrage auditoire",        p.get("score_ancrage_auditoire", 0)),
            ("Cohérence des valeurs",    p.get("score_coherence_valeurs", 0)),
            ("Risque sophistique",       p.get("score_risque_sophistique_rhethorique", 0)),
            ("Profil argumentatif",      p.get("score_profil_argumentatif", 0)),
        ]
        for label, val in scores_affich:
            lignes.append(f"- **{label}** : `{_barre(val)}`")
        lignes.append("")

        # ── Synthèse ──────────────────────────────────────────────────────────
        lignes += ["#### Synthèse rhétorique", ""]
        synthese = _extraire_synthese(analyse_brute)
        for ligne in synthese.split("\n"):
            ligne = ligne.strip()
            if ligne:
                lignes.append(ligne)
        lignes += ["", "---", ""]

    # ── Pied de page ──────────────────────────────────────────────────────────
    lignes += [
        "",
        "*Rapport généré automatiquement par `10_map_perelman_openai.py`.*  ",
        "*Cadre théorique : Perelman & Olbrechts-Tyteca, Traité de l'argumentation (1958).*  ",
        "*Score profil = force×0.30 + ancrage×0.30 + valeurs×0.25 + (1−risque)×0.15*",
    ]

    contenu = "\n".join(lignes)
    chemin_md = output_dir / f"perelman_{timestamp}.md"
    chemin_md.write_text(contenu, encoding="utf-8")
    return chemin_md


# =============================================================================
# TRAITEMENT D'UN PARAGRAPHE
# =============================================================================

def analyser_paragraphe(
    paragraphe: dict,
    llm: "LLMClient",
    scores_09: dict | None = None,
) -> dict | None:
    idx      = paragraphe.get("id", paragraphe.get("index", "?"))
    titre    = paragraphe.get("titre", f"Paragraphe {idx}")
    texte    = paragraphe.get("texte", "")
    passages = paragraphe.get("passages_corpus", paragraphe.get("passages", []))

    if not texte.strip():
        print("  ⚠ Paragraphe sans texte — ignoré.", file=sys.stderr)
        return None

    passages_str = formater_passages(passages, MAX_PASSAGES_PAR_SECTION)
    user_message = construire_user_message(titre, texte, passages_str, scores_09)

    try:
        reponse_brute = llm.generate(SYSTEM_PROMPT, user_message)
    except Exception as e:
        print(f"  ❌ Erreur API : {e}", file=sys.stderr)
        return None

    if not reponse_brute:
        return None

    scores              = extraire_scores(reponse_brute)
    technique_dominante = extraire_technique_dominante(reponse_brute)
    type_auditoire      = extraire_type_auditoire(reponse_brute)
    valeurs             = extraire_valeurs(reponse_brute)
    usages              = extraire_usages_sophistiques(reponse_brute)
    techniques          = extraire_techniques_par_famille(reponse_brute)
    paires              = extraire_paires_dissociation(reponse_brute)

    profil_verif = calculer_profil(scores)
    if abs(scores.get("score_profil_argumentatif", 0.5) - profil_verif) > 0.15:
        print(f"  ℹ Profil corrigé → {profil_verif:.3f}", file=sys.stderr)
        scores["score_profil_argumentatif"] = profil_verif

    return {
        "id"                                   : idx,
        "titre"                                : titre,
        **scores,
        "technique_dominante"                  : technique_dominante,
        "type_auditoire"                       : type_auditoire,
        "techniques_detectees"                 : techniques,
        "paires_dissociation"                  : paires,
        "valeurs_mobilisees"                   : valeurs,
        "usages_sophistiques"                  : usages,
        "analyse"                              : reponse_brute,
        "passages_corpus": [
            {"source": p.get("source",""), "page": p.get("page",""),
             "extrait": p.get("extrait", p.get("texte",""))[:300]}
            for p in passages[:MAX_PASSAGES_PAR_SECTION]
        ],
    }


# =============================================================================
# SCORE GLOBAL + SAUVEGARDE + BILAN
# =============================================================================

def calculer_score_global(paragraphes: list[dict]) -> tuple[float, str]:
    if not paragraphes:
        return 0.0, "indeterminee"
    score = round(sum(s["score_profil_argumentatif"] for s in paragraphes) / len(paragraphes), 3)
    comptage = {}
    for s in paragraphes:
        k = s.get("technique_dominante", "indeterminee")
        comptage[k] = comptage.get(k, 0) + 1
    return score, max(comptage, key=comptage.get)


def sauvegarder(paragraphes: list[dict], score: float, profil: str,
                output_dir: Path, modele: str) -> tuple:
    """
    Sauvegarde le JSON final et génère le rapport Markdown en parallèle.

    Retourne :
        chemin_json : Path — fichier perelman_{timestamp}.json
        chemin_md   : Path — fichier perelman_{timestamp}.md
    """
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin = output_dir / f"perelman_{ts}.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({
            "script": "10_map_perelman_openai", "timestamp": ts,
            "modele_llm": modele, "temperature": TEMPERATURE,
            "score_global": score, "profil_dominant": profil,
            "nb_paragraphes": len(paragraphes), "paragraphes": paragraphes,
        }, f, ensure_ascii=False, indent=2)
    chemin_md = generer_rapport_md(paragraphes, score, profil, ts, modele, output_dir)
    return chemin, chemin_md


def afficher_bilan(paragraphes: list[dict], score_global: float) -> None:
    print("\n" + "═" * 76)
    print(f"  BILAN — 10_map_perelman | Score profil global : {score_global:.3f}")
    print("═" * 76)
    print(f"  {'§':>3}  {'Titre':<28}  {'Force':>6}  {'Ancrag':>6}  "
          f"{'Valeur':>6}  {'Risq.':>6}  {'Profil':>6}  Technique")
    print("─" * 76)
    for s in paragraphes:
        print(f"  {s['id']:>3}  {s['titre'][:28]:<28}  "
              f"{s['score_force_persuasive']:.2f}    {s['score_ancrage_auditoire']:.2f}    "
              f"{s['score_coherence_valeurs']:.2f}    {s['score_risque_sophistique_rhethorique']:.2f}    "
              f"{s['score_profil_argumentatif']:.2f}  {s.get('technique_dominante','?')}")
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
        description="10_map_perelman_openai — Analyse Nouvelle Rhétorique (OpenAI)."
    )
    parser.add_argument("--map_file", type=str, default=None)
    parser.add_argument("--paragraphes", type=int, nargs="+", default=None, metavar="ID")
    parser.add_argument(
        "--avec_argumentation", type=str, default=None, metavar="CHEMIN",
        help="Fichier argumentation_{timestamp}.json produit par 09. "
             "Si fourni, les scores Toulmin/Walton sont inclus dans le prompt.",
    )
    args = parser.parse_args()

    map_dir    = Path(MAP_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    if hasattr(llm, "model"):
        llm.model = OPENAI_LLM_MODEL

    # Chargement du fichier argumentation 09 (optionnel)
    scores_09_par_paragraphe = {}
    if args.avec_argumentation:
        chemin_09 = Path(args.avec_argumentation)
        if chemin_09.exists():
            scores_09_par_paragraphe = charger_argumentation(chemin_09)
            print(f"  Résultats 09 chargés : {chemin_09.name} "
                  f"({len(scores_09_par_paragraphe)} paragraphes)")
        else:
            print(f"  ⚠ Fichier introuvable : {chemin_09}", file=sys.stderr)

    if args.map_file:
        map_file = Path(args.map_file)
        if not map_file.is_absolute():
            map_file = map_dir / map_file
    else:
        print("Aucun fichier map spécifié — recherche du plus récent…")
        map_file = trouver_dernier_map(map_dir)

    print(f"Chargement : {map_file.name}")
    map_data = charger_map(map_file)
    paragraphes = map_data.get("paragraphes", [])

    if args.paragraphes:
        paragraphes = [p for p in paragraphes
                       if p.get("id", p.get("index")) in args.paragraphes]
        if not paragraphes:
            print(f"❌ Aucun paragraphe avec les identifiants {args.paragraphes}.")
            sys.exit(1)

    print(f"  {len(paragraphes)} paragraphe(s) | Modèle : {llm.model} | T° : {TEMPERATURE}\n")

    paragraphes_analyses = []
    for p in paragraphes:
        pid = p.get("id", p.get("index", "?"))
        print(f"§ {pid} — {p.get('titre', p.get('texte','?')[:50])}")
        scores_09 = scores_09_par_paragraphe.get(pid)
        res = analyser_paragraphe(p, llm, scores_09)
        if res:
            print(f"  Profil : {res['score_profil_argumentatif']:.3f} | "
                  f"Technique : {res.get('technique_dominante','?')}")
            paragraphes_analyses.append(res)
        else:
            print("  ⚠ Paragraphe ignoré.")
        time.sleep(PAUSE_INTER_SECTION)
        print()

    if not paragraphes_analyses:
        print("❌ Aucun paragraphe analysé.")
        sys.exit(1)

    score_global, profil = calculer_score_global(paragraphes_analyses)
    chemin_json, chemin_md = sauvegarder(paragraphes_analyses, score_global, profil, output_dir, llm.model)
    afficher_bilan(paragraphes_analyses, score_global)
    print(f"✅ Analyse complète")
    print(f"   JSON : {chemin_json.name}")
    print(f"   MD   : {chemin_md.name}")
    print(f"   Score profil global : {score_global:.3f} | Technique dominante : {profil}\n")


if __name__ == "__main__":
    main()
