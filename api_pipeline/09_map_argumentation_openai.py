"""
09_map_argumentation_openai.py
===============================
Étape 9 du pipeline RAG — Analyse rhétorico-logique du manuscrit.
VERSION OPENAI (API distante, modèles gpt-4.1 / gpt-4.1-mini).

Rôle : Pour chaque section du manuscrit (issue du fichier map produit par
06_map_enrich.py), analyse la structure argumentative selon trois cadres
théoriques complémentaires :

    1. TOULMIN (1958) — Structure logique de l'argument
       Stephen Toulmin, The Uses of Argument (Cambridge UP, 1958).
       Six composantes : claim | grounds | warrant | backing | qualifier | rebuttal

    2. ADAM (1992) — Séquences argumentatives textuelles
       Jean-Michel Adam, Les textes : types et prototypes (Nathan, 1992).
       Quatre moments : thèse antérieure | données/argument | conclusion/thèse | restriction

    3. WALTON (1996) — Schèmes argumentatifs et sophismes
       Douglas Walton, Argumentation Schemes for Presumptive Reasoning (Erlbaum, 1996).
       Schèmes légitimes : autorité | analogie | causal | pragmatique
       Schèmes fallacieux : pente_glissante | generalisation | ad_hominem | homme_de_paille

POSITION DANS LE PIPELINE :
    06_map_enrich.py               → enrich.json  (ou map_{timestamp}.json)
    09_map_argumentation_openai.py → argumentation_{timestamp}.json  (ce script)
    10_map_perelman_openai.py      → perelman_{timestamp}.json

DIFFÉRENCES AVEC LA VERSION LOCALE (09_map_argumentation.py) :
    - Utilise LLMClient depuis 00_config.py (appel via client.generate)
    - Modèle par défaut : gpt-4.1-mini (configurable ci-dessous)
    - Nécessite OPENAI_API_KEY dans .env
    - Coût indicatif : 0.02–0.08 $ par analyse complète avec gpt-4.1-mini

SYSTÈME DE REPRISE SUR INTERRUPTION :
    Le script sauvegarde chaque paragraphe immédiatement après son analyse
    dans un fichier de progression :
        resultats/argumentation_progress_{timestamp}.json

    En cas de coupure réseau, crash ou Ctrl+C, relancez simplement la même
    commande : le script détecte automatiquement la progression, charge les
    paragraphes déjà traités et reprend au premier non encore analysé.

    À la fin de l'analyse complète, le fichier de progression est supprimé
    et remplacé par le fichier final :
        resultats/argumentation_{timestamp}.json

    Pour repartir de zéro en ignorant toute progression :
        python 09_map_argumentation_openai.py --reset

SCORES PRODUITS (tous entre 0.0 et 1.0) :

    score_completude_toulmin [0–1]
        Proportion des six composantes de Toulmin présentes ou inférables.
        0.0 = seule la claim, sans grounds ni warrant.
        1.0 = les six composantes sont identifiables.
        ⚠ Limite : complétude formelle ≠ valeur historiographique de l'argument.

    score_coherence_warrant [0–1]
        Solidité du lien grounds→claim via le warrant (loi de passage).
        0.0 = lien arbitraire ou non justifié.
        1.0 = warrant explicite et étayé par le backing.
        ⚠ Limite : cohérence formelle, pas vérité du warrant.

    score_risque_sophisme [0–1]
        Présence de schèmes fallacieux (Walton).
        0.0 = aucun sophisme. 1.0 = argument principalement fallacieux.

    score_charge_probatoire [0–1]
        Présence de grounds explicites pour chaque claim.
        0.0 = affirmations non étayées. 1.0 = toutes les claims sont étayées.

    score_robustesse_globale [0–1]
        Synthèse pondérée :
            complétude×0.20 + warrant×0.35 + charge×0.25 + (1−sophisme)×0.20

    schema_walton_dominant [str]
        "autorité" | "analogie" | "causal" | "pragmatique" |
        "pente_glissante" | "generalisation" | "ad_hominem" |
        "homme_de_paille" | "aucun"

UTILISATION :
    python 09_map_argumentation_openai.py
    python 09_map_argumentation_openai.py --map_file resultats/enrich.json
    python 09_map_argumentation_openai.py --sections 3 7 12
    python 09_map_argumentation_openai.py --reset
    python 09_map_argumentation_openai.py --texte_seul mon_texte.txt
    python 09_map_argumentation_openai.py --texte_seul mon_bloc.txt --bloc

MODE --texte_seul :
    Analyse un fichier .txt directement, sans corpus associé.
    Le script segmente le texte en paragraphes (séparés par une ligne vide),
    lance l'analyse Toulmin / Adam / Walton sur chacun, et produit le JSON
    habituel avec passages_corpus vide.
    LIMITE : la section "CONFRONTATION CORPUS" du prompt sera vide —
    le LLM ne peut pas vérifier si les grounds sont attestés dans la littérature.
    Toutes les autres analyses (structure logique, schèmes, scores) fonctionnent
    normalement sur le texte seul.
    Le 10 (Perelman) peut ensuite être chaîné sur le JSON produit.

MODE --texte_seul + --bloc :
    Combine les deux arguments pour analyser un fichier .txt en un seul bloc,
    sans aucune segmentation en paragraphes.
    Utile quand le développement argumentatif s'étale sur plusieurs paragraphes
    qui forment une unité rhétorique cohérente (thèse, grounds, warrant distribués).

    Limite recommandée : 1800 mots. Au-delà, le script avertit que la qualité
    de l'analyse Toulmin / Adam / Walton peut se dégrader (le LLM tend à
    simplifier ou à perdre en précision sur les scores). L'analyse est néanmoins
    lancée — l'utilisateur reste libre de passer outre.

    Usage :
        python 09_map_argumentation_openai.py --texte_seul mon_bloc.txt --bloc

Prérequis : fichier .env avec OPENAI_API_KEY=sk-...
"""

# =============================================================================
# PARAMÈTRES — modifier ici sans toucher au reste du script
# =============================================================================

OPENAI_LLM_MODEL      = "gpt-4.1-mini"   # ou "gpt-4.1" pour les analyses finales
TEMPERATURE           = 0.1              # basse pour reproductibilité
MAX_TOKENS            = 2500             # tokens max en sortie par paragraphe
MAX_PASSAGES_PAR_SECTION = 5             # passages corpus dans le prompt
MAP_DIR               = "resultats"      # dossier contenant enrich.json / map_*.json
OUTPUT_DIR            = "resultats"      # dossier de sortie
PAUSE_INTER_SECTION   = 0.5              # pause en secondes entre deux appels API
LIMITE_MOTS_BLOC      = 1800             # seuil d'avertissement en mode --bloc

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
# RÉFÉRENTIEL WALTON
# =============================================================================

SCHEMES_WALTON = {
    "autorité"        : "Argument from Expert Opinion",
    "analogie"        : "Argument from Analogy",
    "causal"          : "Argument from Cause to Effect",
    "pragmatique"     : "Pragmatic Argument",
    "pente_glissante" : "Slippery Slope",
    "generalisation"  : "Hasty Generalization",
    "ad_hominem"      : "Ad Hominem",
    "homme_de_paille" : "Straw Man",
}

SCHEMES_FALLACIEUX = {"pente_glissante", "generalisation", "ad_hominem", "homme_de_paille"}

SYSTEM_PROMPT = (
    "Tu es un expert en rhétorique et logique argumentative, spécialisé dans "
    "l'analyse des textes académiques et historiques. Tu maîtrises les cadres "
    "théoriques de Toulmin, Adam et Walton. Tu analyses les arguments avec "
    "précision et rigueur, en distinguant la structure formelle de la valeur "
    "épistémique du contenu. Tu réponds dans la langue du texte analysé ou "
    "en français si le texte est multilingue."
)

# =============================================================================
# FONCTIONS — chargement du fichier map
# =============================================================================

def trouver_dernier_map(map_dir: Path) -> Path:
    """Cherche map_*.json puis enrich*.json dans map_dir."""
    fichiers = sorted(map_dir.glob("map_*.json"), reverse=True)
    if not fichiers:
        fichiers = sorted(map_dir.glob("enrich*.json"), reverse=True)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier map_*.json ou enrich*.json dans {map_dir}.\n"
            "Lancez d'abord 06_map_enrich.py."
        )
    return fichiers[0]


def charger_map(chemin: Path) -> dict:
    """Charge le fichier map. Accepte les clés 'sections' et 'paragraphes'."""
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    if "sections" not in data and "paragraphes" not in data:
        raise ValueError(
            f"{chemin.name} ne contient ni clé 'sections' ni clé 'paragraphes'."
        )
    return data


def charger_texte_seul(chemin: Path, min_chars: int = 150, segmenter: bool = True) -> dict:
    """
    Charge un fichier .txt et le segmente en paragraphes (mode par défaut)
    ou le traite comme un bloc unique (mode --bloc).

    Utilisé avec --texte_seul pour analyser un texte sans corpus associé.
    En mode segmentation, les paragraphes sont délimités par une ligne vide
    et les blocs trop courts (< min_chars) sont ignorés.
    En mode bloc (segmenter=False), le texte entier est traité comme une seule
    unité argumentative — utile quand le développement rhétorique s'étale sur
    plusieurs paragraphes. Un avertissement est émis au-delà de LIMITE_MOTS_BLOC.

    Args:
        chemin    : Chemin vers le fichier .txt.
        min_chars : Longueur minimale d'un paragraphe (défaut : 150 caractères).
        segmenter : Si False, retourne le texte comme un bloc unique sans découpage.

    Returns:
        Dict au format map standard avec clé "paragraphes" et passages vides.

    Raises:
        FileNotFoundError : Si le fichier est introuvable.
        ValueError        : Si le fichier est vide ou sans paragraphe exploitable.
    """
    if not chemin.exists():
        raise FileNotFoundError(f"Fichier introuvable : {chemin.resolve()}")

    texte = chemin.read_text(encoding="utf-8").strip()
    if not texte:
        raise ValueError(f"Le fichier {chemin} est vide.")

    # ── Mode bloc : pas de segmentation ──────────────────────────────────────
    if not segmenter:
        nb_mots = len(texte.split())
        print(f"  Mode bloc : {nb_mots} mots chargés depuis {chemin.name}")
        print("  ⚠ Pas de corpus associé — la section CONFRONTATION CORPUS sera vide.")
        if nb_mots > LIMITE_MOTS_BLOC:
            print(
                f"\n⚠  AVERTISSEMENT : le bloc contient {nb_mots} mots "
                f"(seuil recommandé : {LIMITE_MOTS_BLOC}).\n"
                "   Au-delà de ce seuil, la précision de l'analyse Toulmin / Adam / Walton\n"
                "   peut se dégrader (simplification ou perte de finesse sur les scores).\n"
                "   Appuyez sur Entrée pour continuer malgré tout, ou Ctrl+C pour annuler."
            )
            input()
        return {
            "source":      str(chemin),
            "paragraphes": [
                {
                    "index":    1,
                    "texte":    texte,
                    "nb_chars": len(texte),
                    "passages": [],
                }
            ],
        }

    # ── Mode segmentation : découpage en paragraphes ─────────────────────────
    blocs = re.split(r"\n\s*\n", texte)
    paragraphes = []
    index = 1
    for bloc in blocs:
        bloc = bloc.strip()
        if len(bloc) < min_chars:
            continue
        paragraphes.append({
            "index":    index,
            "texte":    bloc,
            "nb_chars": len(bloc),
            "passages": [],   # pas de corpus — confrontation corpus désactivée
        })
        index += 1

    if not paragraphes:
        raise ValueError(
            f"Aucun paragraphe exploitable (>{min_chars} caractères) dans {chemin}."
        )

    print(f"  Mode texte seul : {len(paragraphes)} paragraphe(s) chargé(s) depuis {chemin.name}")
    print("  ⚠ Pas de corpus associé — la section CONFRONTATION CORPUS sera vide.")

    return {
        "source":      str(chemin),
        "paragraphes": paragraphes,
    }


# =============================================================================
# FONCTIONS — système de reprise sur interruption
# =============================================================================

def trouver_fichier_progression(output_dir: Path) -> Path | None:
    """
    Retourne le fichier argumentation_progress_*.json le plus récent,
    ou None si aucun fichier de progression n'existe.
    """
    fichiers = sorted(output_dir.glob("argumentation_progress_*.json"), reverse=True)
    return fichiers[0] if fichiers else None


def charger_progression(chemin: Path) -> tuple[str, dict]:
    """
    Charge un fichier de progression.

    Retourne :
        timestamp    : str  — timestamp de la session originale, réutilisé
                              pour nommer le fichier final avec le même identifiant
        deja_traites : dict — {id_paragraphe: résultat} des paragraphes déjà analysés
    """
    with open(chemin, encoding="utf-8") as f:
        data = json.load(f)
    timestamp    = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    deja_traites = {s["id"]: s for s in data.get("sections", [])}
    print(f"  ↩  Reprise : {chemin.name}")
    print(f"     Paragraphes déjà traités : {len(deja_traites)}")
    return timestamp, deja_traites


def sauvegarder_progression(
    sections: list[dict],
    output_dir: Path,
    timestamp: str,
    modele: str,
) -> None:
    """
    Écrit le fichier de progression après chaque paragraphe analysé.

    Le fichier est écrasé à chaque appel (même nom, même timestamp).
    Il a la même structure que le fichier final — il peut être utilisé
    directement par le script 10 en cas d'interruption définitive.
    """
    chemin = output_dir / f"argumentation_progress_{timestamp}.json"
    with open(chemin, "w", encoding="utf-8") as f:
        json.dump({
            "script"     : "09_map_argumentation_openai",
            "timestamp"  : timestamp,
            "modele_llm" : modele,
            "temperature": TEMPERATURE,
            "en_cours"   : True,          # marqueur : analyse non terminée
            "nb_sections": len(sections),
            "sections"   : sections,
        }, f, ensure_ascii=False, indent=2)


def finaliser(
    sections: list[dict],
    score: float,
    profil: str,
    output_dir: Path,
    timestamp: str,
    modele: str,
    chemin_progress: Path | None,
) -> tuple:
    """
    Sauvegarde le fichier JSON final, produit le rapport Markdown parallèle,
    et supprime le fichier de progression.

    Le timestamp est celui de la session originale, même si elle a été
    interrompue et reprise — pour faciliter le chaînage avec le script 10.

    Retourne :
        chemin_json : Path — fichier argumentation_{timestamp}.json
        chemin_md   : Path — fichier argumentation_{timestamp}.md
    """
    chemin_final = output_dir / f"argumentation_{timestamp}.json"
    with open(chemin_final, "w", encoding="utf-8") as f:
        json.dump({
            "script"         : "09_map_argumentation_openai",
            "timestamp"      : timestamp,
            "modele_llm"     : modele,
            "temperature"    : TEMPERATURE,
            "en_cours"       : False,
            "score_global"   : score,
            "profil_dominant": profil,
            "nb_sections"    : len(sections),
            "sections"       : sections,
        }, f, ensure_ascii=False, indent=2)

    # Rapport Markdown en parallèle
    chemin_md = generer_rapport_md(sections, score, profil, timestamp, modele, output_dir)

    # Suppression du fichier de progression
    if chemin_progress and chemin_progress.exists():
        chemin_progress.unlink()
        print(f"  Fichier de progression supprimé : {chemin_progress.name}")

    return chemin_final, chemin_md


# =============================================================================
# FONCTIONS — construction du prompt
# =============================================================================

def formater_passages(passages: list[dict], max_passages: int) -> str:
    """
    Formate les passages corpus pour le prompt.
    Compatible avec deux formats :
      - {source, page, extrait/texte} : inclut le texte
      - {source, page, distance}      : référence seule (format enrich.json)
    """
    lignes = []
    for i, p in enumerate(passages[:max_passages], 1):
        source = p.get("ref_courte", p.get("source", "source inconnue"))
        page   = p.get("page", "?")
        texte  = p.get("extrait", p.get("texte", "")).strip()
        if texte:
            lignes.append(f"[Passage {i} — {source}, p. {page}]\n{texte}")
        else:
            lignes.append(f"[Passage {i} — {source}, p. {page}]")
    return "\n\n".join(lignes) if lignes else "(aucun passage corpus disponible)"


def construire_user_message(titre: str, texte: str, passages_str: str) -> str:
    return f"""Analyse le passage suivant selon les cadres Toulmin, Adam et Walton,
puis attribue les scores demandés.

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
Identifie les six composantes (si absentes : ABSENT) :
  CLAIM     : thèse ou conclusion principale
  GROUNDS   : données, faits ou preuves invoquées
  WARRANT   : règle autorisant le passage de GROUNDS à CLAIM
  BACKING   : autorité ou preuve légitimant le WARRANT
  QUALIFIER : nuances ou modalités de la CLAIM
  REBUTTAL  : cas d'exception ou objections anticipées

━━━ 2. ANALYSE ADAM ━━━
Quatre moments de la séquence argumentative :
  THÈSE ANTÉRIEURE : position de départ ou thèse adverse implicite
  DONNÉES          : arguments et preuves mobilisés
  CONCLUSION/THÈSE : nouvelle thèse défendue
  RESTRICTION      : concessions ou limites

━━━ 3. ANALYSE WALTON ━━━
Schèmes présents parmi :
  Légitimes  : autorité | analogie | causal | pragmatique
  Fallacieux : pente_glissante | generalisation | ad_hominem | homme_de_paille
Cite le passage exact pour chaque schème détecté.
SCHEMA_DOMINANT: <schème>
SOPHISMES: <schème1>, <schème2>  (ou AUCUN)

━━━ 4. CONFRONTATION CORPUS ━━━
Les GROUNDS du manuscrit sont-ils attestés dans les passages corpus ?
Pour chaque ground : ATTESTÉ | ABSENT DU CORPUS | CONTREDIT PAR LE CORPUS

━━━ 5. SCORES (format strict — ne pas modifier) ━━━
  SCORE_COMPLETUDE_TOULMIN: X/10
  SCORE_COHERENCE_WARRANT: X/10
  SCORE_RISQUE_SOPHISME: X/10
  SCORE_CHARGE_PROBATOIRE: X/10
  SCORE_ROBUSTESSE_GLOBALE: X/10  ← complétude×0.20 + warrant×0.35 + charge×0.25 + (10−sophisme)×0.20, divisé par 10

━━━ 6. SYNTHÈSE ━━━
En 3–4 phrases : évaluation argumentative globale, point fort, point faible, révision prioritaire."""


# =============================================================================
# FONCTIONS — extraction
# =============================================================================

NOMS_COURTS_SCORES = {
    "score_completude_toulmin": ["completude", "complétude", "completude_toulmin"],
    "score_coherence_warrant" : ["warrant", "coherence_warrant", "cohérence_warrant"],
    "score_risque_sophisme"   : ["sophisme", "risque_sophisme", "risque"],
    "score_charge_probatoire" : ["probatoire", "charge_probatoire", "charge"],
    "score_robustesse_globale": ["robustesse", "robustesse_globale"],
}

PATTERNS_SCORES = {
    "score_completude_toulmin": r"SCORE_COMPLETUDE_TOULMIN\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_coherence_warrant" : r"SCORE_COHERENCE_WARRANT\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_risque_sophisme"   : r"SCORE_RISQUE_SOPHISME\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_charge_probatoire" : r"SCORE_CHARGE_PROBATOIRE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
    "score_robustesse_globale": r"SCORE_ROBUSTESSE_GLOBALE\s*:\s*(\d+(?:[.,]\d+)?)\s*/\s*10",
}

VALEUR_NEUTRE = 0.5


def _normaliser_score(v: float, avec_denominateur: bool) -> float:
    """
    Normalise une valeur brute LLM vers [0, 1].

    avec_denominateur=True  : valeur lue comme N/10 → diviser par 10.
    avec_denominateur=False : ambigu — règle :
        - entier (1, 8, 10…) → sur 10, diviser par 10.
        - décimal > 1 (8.1, 7.5…) → sur 10, diviser par 10.
        - décimal ≤ 1 (0.8, 0.75…) → déjà normalisé, garder tel quel.
    Cette règle est conservatrice : un LLM qui écrit "1" sans /10 voulait
    presque certainement dire "1/10", pas "score parfait de 1.0".
    """
    if avec_denominateur or v == int(v) or v > 1.0:
        return round(max(0.0, min(10.0, v)) / 10.0, 3)
    return round(max(0.0, min(1.0, v)), 3)


def extraire_scores(texte_llm: str) -> dict:
    """
    Extrait les cinq scores Toulmin depuis la réponse LLM brute.

    Quatre niveaux de robustesse, du plus strict au plus permissif :

    Niveau 1 — Format strict : SCORE_X: N/10
        Accepte point ou virgule comme séparateur décimal.
        Ex : SCORE_COMPLETUDE_TOULMIN: 8/10  |  SCORE_RISQUE_SOPHISME: 7,5/10

    Niveau 2 — Espaces autour du / : SCORE_X: N / 10
        Ex : SCORE_COHERENCE_WARRANT: 9 / 10

    Niveau 3 — Nom court sans préfixe SCORE_
        Le LLM peut écrire "complétude : 8" ou "robustesse = 7.5"
        sans /10. Normalisation : entier ou décimal > 1 → sur 10 ;
        décimal ≤ 1 → déjà normalisé.

    Niveau 4 — Tableau Markdown
        | label | valeur | — même règle de normalisation que le niveau 3.

    Si aucun niveau ne trouve le score : retourne VALEUR_NEUTRE (0.5)
    avec un avertissement stderr. 0.5 est une valeur neutre explicite,
    pas un score — à ne pas interpréter comme "argument moyen".
    """
    scores = {}

    for nom, pattern in PATTERNS_SCORES.items():
        # Niveau 1 + 2 — format strict (avec /10 explicite)
        m = re.search(pattern, texte_llm, re.IGNORECASE)
        if m:
            v = float(m.group(1).replace(",", "."))
            scores[nom] = _normaliser_score(v, avec_denominateur=True)
            continue

        # Niveau 3 — nom court, avec ou sans /10
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
            print(f"  ⚠ Score non trouvé : {nom} → {VALEUR_NEUTRE}",
                  file=sys.stderr)
            scores[nom] = VALEUR_NEUTRE

    return scores


def extraire_schema_walton(texte_llm: str) -> tuple[str, list[str]]:
    m = re.search(r"SCHEMA_DOMINANT\s*:\s*(\w+)", texte_llm, re.IGNORECASE)
    schema = m.group(1).lower() if m else "aucun"
    if schema not in SCHEMES_WALTON:
        schema = "aucun"
    m2 = re.search(r"SOPHISMES\s*:\s*(.+)", texte_llm, re.IGNORECASE)
    sophismes = []
    if m2 and m2.group(1).strip().upper() != "AUCUN":
        sophismes = [s.strip().lower() for s in re.split(r"[,;]", m2.group(1))
                     if s.strip().lower() in SCHEMES_FALLACIEUX]
    return schema, sophismes


def extraire_composantes_toulmin(texte_llm: str) -> dict:
    composantes = {}
    for comp in ["CLAIM", "GROUNDS", "WARRANT", "BACKING", "QUALIFIER", "REBUTTAL"]:
        m = re.search(
            rf"{comp}\s*:\s*(.+?)(?=\n\s*(?:CLAIM|GROUNDS|WARRANT|BACKING|QUALIFIER|REBUTTAL|━|$))",
            texte_llm, re.IGNORECASE | re.DOTALL,
        )
        if m:
            val = m.group(1).strip()
            composantes[comp.lower()] = val if val.upper() != "ABSENT" else "ABSENT"
        else:
            composantes[comp.lower()] = "ABSENT"
    return composantes


def extraire_sequence_adam(texte_llm: str) -> dict:
    champs = {
        "these_anterieure": r"THÈSE ANTÉRIEURE\s*:\s*(.+?)(?=\n\s*(?:DONNÉES|CONCLUSION|RESTRICTION|━|$))",
        "donnees"         : r"DONNÉES\s*:\s*(.+?)(?=\n\s*(?:CONCLUSION|RESTRICTION|━|$))",
        "conclusion"      : r"CONCLUSION(?:/THÈSE)?\s*:\s*(.+?)(?=\n\s*(?:RESTRICTION|━|$))",
        "restriction"     : r"RESTRICTION\s*:\s*(.+?)(?=\n\s*(?:━|$))",
    }
    return {
        cle: (re.search(pat, texte_llm, re.IGNORECASE | re.DOTALL).group(1).strip()
              if re.search(pat, texte_llm, re.IGNORECASE | re.DOTALL) else "non identifié")
        for cle, pat in champs.items()
    }


def calculer_robustesse(scores: dict) -> float:
    return round(
        scores.get("score_completude_toulmin", 0.5) * 0.20
        + scores.get("score_coherence_warrant", 0.5) * 0.35
        + scores.get("score_charge_probatoire", 0.5) * 0.25
        + (1.0 - scores.get("score_risque_sophisme", 0.5)) * 0.20,
        3,
    )


# =============================================================================
# GÉNÉRATION DU RAPPORT MARKDOWN
# =============================================================================

def _barre(valeur: float, largeur: int = 20) -> str:
    """Barre ASCII proportionnelle. Ex : [████████░░░░░░░░░░░░] 0.42"""
    rempli = round(valeur * largeur)
    return f"[{'█' * rempli}{'░' * (largeur - rempli)}] {valeur:.2f}"


def generer_rapport_md(
    sections: list[dict],
    score_global: float,
    profil: str,
    timestamp: str,
    modele: str,
    output_dir: Path,
) -> Path:
    """
    Génère argumentation_{timestamp}.md — rapport lisible par section.

    Structure par section :
      - Extrait du texte analysé
      - Composantes Toulmin (claim, grounds, warrant, backing, qualifier, rebuttal)
      - Séquence Adam (thèse antérieure, données, conclusion, restriction)
      - Schème Walton dominant + sophismes détectés avec citations
      - Scores numériques avec barres ASCII
      - Synthèse LLM (3–4 phrases extraites de l'analyse brute)

    Le fichier est produit en parallèle du JSON dans le même OUTPUT_DIR.
    """

    def _extraire_synthese(analyse_brute: str) -> str:
        """Extrait le bloc SYNTHÈSE de la réponse LLM brute."""
        m = re.search(
            r"(?:synthèse|SYNTHÈSE)[^\n]*\n(.+?)(?:\n\s*━|$)",
            analyse_brute,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            return m.group(1).strip()
        # Fallback : derniers 600 caractères si le marqueur est absent
        return analyse_brute.strip()[-600:] if analyse_brute else "(non disponible)"

    lignes = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lignes += [
        f"# Rapport d'analyse argumentative",
        f"",
        f"**Script** : `09_map_argumentation_openai`  ",
        f"**Timestamp** : {timestamp}  ",
        f"**Modèle** : {modele}  ",
        f"**Température** : {TEMPERATURE}  ",
        f"**Sections analysées** : {len(sections)}  ",
        f"",
        f"---",
        f"",
        f"## Bilan global",
        f"",
        f"| Métrique | Valeur |",
        f"|---|---|",
        f"| Score de robustesse global | **{score_global:.3f}** |",
        f"| Schème Walton dominant | `{profil}` |",
        f"",
    ]

    # Tableau récapitulatif
    lignes += [
        "| § | Titre | Complét. | Warrant | Probat. | Sophisme | Robust. | Schème |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in sections:
        sophisme_flag = "⚠ " + ", ".join(s.get("sophismes_detectes", [])) if s.get("sophismes_detectes") else "—"
        lignes.append(
            f"| {s['id']} "
            f"| {s.get('titre', '')[:35]} "
            f"| {s['score_completude_toulmin']:.2f} "
            f"| {s['score_coherence_warrant']:.2f} "
            f"| {s['score_charge_probatoire']:.2f} "
            f"| {sophisme_flag} "
            f"| **{s['score_robustesse_globale']:.2f}** "
            f"| `{s.get('schema_walton_dominant','aucun')}` |"
        )
    lignes += ["", "---", ""]

    # ── Sections détaillées ───────────────────────────────────────────────────
    lignes.append("## Analyse par section\n")

    for s in sections:
        idx   = s.get("id", "?")
        titre = s.get("titre", f"Paragraphe {idx}")
        texte = s.get("analyse", "")  # texte brut LLM

        # Extrait du texte manuscrit (200 premiers caractères)
        # Récupéré depuis la clé analyse (LLM) ou absent si non stocké
        extrait_texte = ""
        for p in s.get("passages_corpus", []):
            if p.get("extrait"):
                extrait_texte = p["extrait"][:200]
                break

        lignes += [
            f"### § {idx} — {titre}",
            f"",
        ]

        if extrait_texte:
            lignes += [
                f"> *{extrait_texte}{'…' if len(extrait_texte) == 200 else ''}*",
                f"",
            ]

        # ── Composantes Toulmin ───────────────────────────────────────────────
        lignes += ["#### Analyse Toulmin", ""]
        comp = s.get("composantes_toulmin", {})
        etiquettes = {
            "claim"   : "CLAIM     — thèse ou conclusion",
            "grounds" : "GROUNDS   — données / preuves",
            "warrant" : "WARRANT   — loi de passage",
            "backing" : "BACKING   — appui du warrant",
            "qualifier": "QUALIFIER — nuances / modalités",
            "rebuttal": "REBUTTAL  — objections anticipées",
        }
        for cle, label in etiquettes.items():
            valeur = comp.get(cle, "ABSENT")
            prefixe = "✗" if valeur == "ABSENT" else "✔"
            lignes.append(f"- **{prefixe} {label}** : {valeur}")
        lignes.append("")

        # ── Séquence Adam ─────────────────────────────────────────────────────
        lignes += ["#### Séquence Adam", ""]
        seq = s.get("sequence_adam", {})
        adam_etiquettes = {
            "these_anterieure": "Thèse antérieure",
            "donnees"         : "Données / arguments",
            "conclusion"      : "Conclusion / thèse",
            "restriction"     : "Restriction / concession",
        }
        for cle, label in adam_etiquettes.items():
            valeur = seq.get(cle, "non identifié")
            lignes.append(f"- **{label}** : {valeur}")
        lignes.append("")

        # ── Walton ────────────────────────────────────────────────────────────
        lignes += ["#### Schème Walton et sophismes", ""]
        schema = s.get("schema_walton_dominant", "aucun")
        desc   = SCHEMES_WALTON.get(schema, "—")
        lignes.append(f"- **Schème dominant** : `{schema}` — *{desc}*")
        sophismes = s.get("sophismes_detectes", [])
        if sophismes:
            lignes.append(f"- **⚠ Sophismes détectés** : {', '.join(f'`{x}`' for x in sophismes)}")
        else:
            lignes.append("- **Sophismes détectés** : aucun")
        lignes.append("")

        # ── Scores ────────────────────────────────────────────────────────────
        lignes += ["#### Scores", ""]
        scores_affich = [
            ("Complétude Toulmin",  s.get("score_completude_toulmin", 0)),
            ("Cohérence warrant",   s.get("score_coherence_warrant", 0)),
            ("Charge probatoire",   s.get("score_charge_probatoire", 0)),
            ("Risque sophisme",     s.get("score_risque_sophisme", 0)),
            ("Robustesse globale",  s.get("score_robustesse_globale", 0)),
        ]
        for label, val in scores_affich:
            lignes.append(f"- **{label}** : `{_barre(val)}`")
        lignes.append("")

        # ── Synthèse ──────────────────────────────────────────────────────────
        lignes += ["#### Synthèse", ""]
        synthese = _extraire_synthese(texte)
        for ligne in synthese.split("\n"):
            ligne = ligne.strip()
            if ligne:
                lignes.append(ligne)
        lignes += ["", "---", ""]

    # ── Pied de page ──────────────────────────────────────────────────────────
    lignes += [
        "",
        "*Rapport généré automatiquement par `09_map_argumentation_openai.py`.*  ",
        f"*Cadres théoriques : Toulmin (1958), Adam (1992), Walton (1996).*  ",
        f"*Score robustesse = complétude×0.20 + warrant×0.35 + charge×0.25 + (1−sophisme)×0.20*",
    ]

    contenu = "\n".join(lignes)
    chemin_md = output_dir / f"argumentation_{timestamp}.md"
    chemin_md.write_text(contenu, encoding="utf-8")
    return chemin_md


# =============================================================================
# TRAITEMENT D'UNE SECTION
# =============================================================================

def analyser_section(section: dict, llm: "LLMClient") -> dict | None:
    """
    Analyse un paragraphe via le LLM et retourne le résultat structuré.
    Compatible format map_*.json (id/titre/texte/passages_corpus)
    et format enrich.json (index/texte/passages).
    """
    idx      = section.get("id", section.get("index", "?"))
    titre    = section.get("titre", f"Paragraphe {idx}")
    texte    = section.get("texte", "")
    passages = section.get("passages_corpus", section.get("passages", []))

    if not texte.strip():
        print("  ⚠ Section sans texte — ignorée.", file=sys.stderr)
        return None

    passages_str = formater_passages(passages, MAX_PASSAGES_PAR_SECTION)
    user_message = construire_user_message(titre, texte, passages_str)

    try:
        reponse_brute = llm.generate(SYSTEM_PROMPT, user_message)
    except Exception as e:
        print(f"  ❌ Erreur API : {e}", file=sys.stderr)
        return None

    if not reponse_brute:
        return None

    scores            = extraire_scores(reponse_brute)
    schema, sophismes = extraire_schema_walton(reponse_brute)
    composantes       = extraire_composantes_toulmin(reponse_brute)
    sequence          = extraire_sequence_adam(reponse_brute)

    robustesse_verif = calculer_robustesse(scores)
    if abs(scores.get("score_robustesse_globale", 0.5) - robustesse_verif) > 0.15:
        print(f"  ℹ Robustesse corrigée → {robustesse_verif:.3f}", file=sys.stderr)
        scores["score_robustesse_globale"] = robustesse_verif

    return {
        "id"                     : idx,
        "titre"                  : titre,
        **scores,
        "schema_walton_dominant" : schema,
        "sophismes_detectes"     : sophismes,
        "composantes_toulmin"    : composantes,
        "sequence_adam"          : sequence,
        "analyse"                : reponse_brute,
        "passages_corpus": [
            {"source": p.get("source",""), "page": p.get("page",""),
             "extrait": p.get("extrait", p.get("texte",""))[:300]}
            for p in passages[:MAX_PASSAGES_PAR_SECTION]
        ],
    }


# =============================================================================
# SCORE GLOBAL + BILAN
# =============================================================================

def calculer_score_global(sections: list[dict]) -> tuple[float, str]:
    if not sections:
        return 0.0, "aucun"
    score = round(sum(s["score_robustesse_globale"] for s in sections) / len(sections), 3)
    comptage = {}
    for s in sections:
        k = s.get("schema_walton_dominant", "aucun")
        comptage[k] = comptage.get(k, 0) + 1
    return score, max(comptage, key=comptage.get)


def afficher_bilan(sections: list[dict], score_global: float) -> None:
    print("\n" + "═" * 72)
    print(f"  BILAN — 09_map_argumentation | Score global : {score_global:.3f}")
    print("═" * 72)
    print(f"  {'§':>3}  {'Titre':<28}  {'Complt':>6}  {'Warrt':>6}  "
          f"{'Probat':>6}  {'Sophis':>6}  {'Robust':>6}  Schème")
    print("─" * 72)
    for s in sections:
        print(f"  {s['id']:>3}  {s['titre'][:28]:<28}  "
              f"{s['score_completude_toulmin']:.2f}    {s['score_coherence_warrant']:.2f}    "
              f"{s['score_charge_probatoire']:.2f}    {s['score_risque_sophisme']:.2f}    "
              f"{s['score_robustesse_globale']:.2f}  {s.get('schema_walton_dominant','aucun')}")
        if s.get("sophismes_detectes"):
            print(f"       ⚠ Sophismes : {', '.join(s['sophismes_detectes'])}")
    print("═" * 72 + "\n")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="09_map_argumentation_openai — Analyse Toulmin + Adam + Walton (OpenAI)."
    )
    parser.add_argument("--map_file", type=str, default=None,
                        help="Fichier map JSON (enrich.json ou map_*.json). "
                             "Si absent : prend le plus récent dans MAP_DIR.")
    parser.add_argument("--texte_seul", type=str, default=None, metavar="FICHIER",
                        help="Fichier .txt à analyser directement, sans corpus associé. "
                             "Incompatible avec --map_file.")
    parser.add_argument("--sections", type=int, nargs="+", default=None, metavar="ID",
                        help="Identifiants des paragraphes à analyser. Si absent : tous.")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore toute progression antérieure et repart de zéro.")
    parser.add_argument(
        "--bloc",
        action="store_true",
        help="Avec --texte_seul : traite le fichier comme un bloc unique sans découpage "
             "en paragraphes. Utile pour les développements argumentatifs multi-paragraphes "
             f"(avertissement au-delà de {LIMITE_MOTS_BLOC} mots).",
    )
    args = parser.parse_args()

    if args.map_file and args.texte_seul:
        print("❌ --map_file et --texte_seul sont incompatibles. Choisissez l'un ou l'autre.")
        sys.exit(1)

    if args.bloc and not args.texte_seul:
        print("❌ --bloc ne peut être utilisé que conjointement avec --texte_seul.")
        sys.exit(1)

    map_dir    = Path(MAP_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLMClient()
    if hasattr(llm, "model"):
        llm.model = OPENAI_LLM_MODEL

    # ── Chargement du fichier map ou du texte seul ────────────────────────────
    if args.texte_seul:
        try:
            map_data = charger_texte_seul(Path(args.texte_seul), segmenter=not args.bloc)
        except (FileNotFoundError, ValueError) as e:
            print(f"❌ {e}")
            sys.exit(1)
    elif args.map_file:
        map_file = Path(args.map_file)
        if not map_file.is_absolute():
            map_file = map_dir / map_file
        print(f"Chargement : {map_file.name}")
        map_data = charger_map(map_file)
    else:
        print("Aucun fichier map spécifié — recherche du plus récent…")
        map_file = trouver_dernier_map(map_dir)
        print(f"Chargement : {map_file.name}")
        map_data = charger_map(map_file)

    sections = map_data.get("sections", map_data.get("paragraphes", []))

    if args.sections:
        sections = [s for s in sections
                    if s.get("id", s.get("index")) in args.sections]
        if not sections:
            print(f"❌ Aucune section avec les identifiants {args.sections}.")
            sys.exit(1)

    # ── Détection de la progression antérieure ────────────────────────────────
    deja_traites    = {}
    chemin_progress = None
    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.reset:
        print("Option --reset : démarrage forcé depuis le début.")
        for f in output_dir.glob("argumentation_progress_*.json"):
            f.unlink()
            print(f"  Progression supprimée : {f.name}")
    else:
        chemin_progress = trouver_fichier_progression(output_dir)
        if chemin_progress:
            timestamp, deja_traites = charger_progression(chemin_progress)
        else:
            print("Aucune progression antérieure — démarrage complet.")

    nb_total  = len(sections)
    nb_sautes = sum(1 for s in sections
                    if s.get("id", s.get("index")) in deja_traites)
    print(f"\n  {nb_total} paragraphe(s) au total | "
          f"{nb_sautes} déjà traité(s) | "
          f"{nb_total - nb_sautes} à analyser")
    print(f"  Modèle : {llm.model} | T° : {TEMPERATURE}\n")

    # ── Boucle principale ─────────────────────────────────────────────────────
    # Initialise la liste avec les résultats déjà acquis
    sections_analysees = list(deja_traites.values())

    try:
        for sec in sections:
            sid = sec.get("id", sec.get("index", "?"))

            # Paragraphe déjà traité → on saute sans appel API
            if sid in deja_traites:
                print(f"§ {sid} — déjà traité ✓")
                continue

            print(f"§ {sid} — {sec.get('titre', f'Paragraphe {sid}')}")
            res = analyser_section(sec, llm)

            if res:
                print(f"  Robustesse : {res['score_robustesse_globale']:.3f} | "
                      f"Schème : {res.get('schema_walton_dominant','aucun')}")
                sections_analysees.append(res)
                deja_traites[sid] = res
                # Sauvegarde immédiate après chaque paragraphe réussi
                sauvegarder_progression(
                    sections_analysees, output_dir, timestamp, llm.model
                )
            else:
                print("  ⚠ Section ignorée.")

            time.sleep(PAUSE_INTER_SECTION)
            print()

    except KeyboardInterrupt:
        # Ctrl+C : la progression est déjà sauvegardée paragraphe par paragraphe
        print(f"\n\n⚠ Interruption clavier.")
        print(f"  {len(sections_analysees)} paragraphe(s) sauvegardés dans :")
        print(f"  {output_dir}/argumentation_progress_{timestamp}.json")
        print("  Relancez le script pour reprendre automatiquement.")
        sys.exit(0)

    # ── Finalisation ──────────────────────────────────────────────────────────
    if not sections_analysees:
        print("❌ Aucune section analysée.")
        sys.exit(1)

    score_global, profil = calculer_score_global(sections_analysees)
    chemin_final, chemin_md = finaliser(
        sections_analysees, score_global, profil,
        output_dir, timestamp, llm.model, chemin_progress,
    )

    afficher_bilan(sections_analysees, score_global)
    print(f"✅ Analyse complète")
    print(f"   JSON : {chemin_final.name}")
    print(f"   MD   : {chemin_md.name}")
    print(f"   Score global : {score_global:.3f} | Schème dominant : {profil}\n")


if __name__ == "__main__":
    main()
