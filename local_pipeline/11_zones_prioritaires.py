#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11_zones_prioritaires.py
========================
Étape 11 du pipeline RAG — Cartographie des zones prioritaires.

Rôle : Lit les JSON produits par 09_map_argumentation_openai.py et
10_map_perelman_openai.py, calcule six signaux croisés entre les deux
analyses, et désigne les paragraphes qui méritent une analyse approfondie
avec les scripts A (Toulmin), B (Perelman) ou C (RST) sur segment défini.

CE QUE CE SCRIPT FAIT ET NE FAIT PAS
─────────────────────────────────────
Il FAIT : signaler des zones à relire, en croisant deux analyses imparfaites
pour produire un signal utile à l'orientation du travail de l'historien.

Il NE FAIT PAS : produire des métriques argumentatives fiables au niveau du
paragraphe. Les scripts 09 et 10 opèrent sur des unités trop petites pour
que leurs scores individuels soient exploitables comme tels. Leur valeur 
éventuelle est dans la distribution agrégée (portrait rhétorique du texte entier)
 et dans les signaux de localisation produits ici.

Cette distinction est fondamentale et doit être assumée dans toute
présentation du dispositif.

AUCUN APPEL LLM — coût zéro.

LES SIX SIGNAUX CROISÉS
───────────────────────
  A  rhéto_sans_preuve     : robustesse 09 < SEUIL_ROBUST_BAS
                             ET profil Perelman 10 ≥ SEUIL_PROFIL_HAUT
                             → le paragraphe argumente efficacement sans étayer.
                             Candidat prioritaire pour script A.

  B  force_sans_grounds    : charge probatoire 09 ≤ SEUIL_PROBAT_BAS
                             ET force persuasive 10 ≥ SEUIL_FORCE_HAUT
                             → rhétorique forte, fondement logique absent.
                             Candidat pour script A + B combinés.

  C  double_sophisme       : sophisme détecté en 09 ET risque sophistique
                             10 ≥ SEUIL_RISQUE_SOPH
                             → deux cadres convergent sur une anomalie.
                             Candidat urgent pour script A.

  D  09_muet_10_parle      : robustesse 09 = 0.50 (valeur neutre = extraction
                             échouée) ET profil 10 > SEUIL_PROFIL_MUET
                             → le 09 n'a pas vu d'argument mais le 10 voit
                             une rhétorique. Souvent : discours rapporté,
                             liste bibliographique, ou conclusion narrative.
                             À examiner pour confirmer si c'est voulu.

  E  autorite_disciplinaire: ancrage auditoire 10 ≥ SEUIL_ANCRAGE_HAUT
                             ET charge probatoire 09 ≤ SEUIL_PROBAT_BAS
                             ET technique 10 contient "autorité"
                             → argument d'autorité disciplinaire sans grounds
                             explicites. Fréquent et souvent non problématique,
                             mais mérite attention dans les sections clés.

  F  coherence_axiologique : valeurs 10 ≥ SEUIL_VALEURS_HAUT
                             ET risque sophistique 10 ≤ SEUIL_RISQUE_BAS
                             ET profil 10 ≥ SEUIL_PROFIL_AXIOLOGIQUE
                             → passage à forte cohérence de valeurs, bien
                             ancré rhétoriquement. Signal positif : zones
                             rhétoriquement solides du texte.

PRIORITÉS
─────────
  Haute  : 2 signaux ou plus (dont au moins un A/B/C)
  Moyenne: 1 signal A, B ou C
  Info   : signal F seul (cohérence axiologique — signal positif)

POSITION DANS LE PIPELINE
──────────────────────────
  09_map_argumentation_openai.py → argumentation_{ts}.json
  10_map_perelman_openai.py      → perelman_{ts}.json
  11_zones_prioritaires.py       → zones_{ts}.md  (ce script)
  → puis scripts A, B, C sur les zones désignées

UTILISATION
───────────
  python 11_zones_prioritaires.py
  python 11_zones_prioritaires.py --arg09 resultats/argumentation_X.json \\
                                   --per10 resultats/perelman_X.json
  python 11_zones_prioritaires.py --seuil_robust 0.60
"""

# =============================================================================
# PARAMÈTRES — modifier ici
# =============================================================================

# Dossier où chercher les JSON 09 et 10
JSON_DIR   = "outputs"
OUTPUT_DIR = "outputs"

# ── Seuils des signaux ──────────────────────────────────────────────────────
# Signal A — rhéto_sans_preuve
SEUIL_ROBUST_BAS    = 0.55   # robustesse 09 en-dessous de ce seuil
SEUIL_PROFIL_HAUT   = 0.70   # profil Perelman 10 au-dessus de ce seuil

# Signal B — force_sans_grounds
SEUIL_PROBAT_BAS    = 0.40   # charge probatoire 09 en-dessous
SEUIL_FORCE_HAUT    = 0.65   # force persuasive 10 au-dessus

# Signal C — double_sophisme
SEUIL_RISQUE_SOPH   = 0.40   # risque sophistique 10 au-dessus

# Signal D — 09_muet_10_parle
SEUIL_PROFIL_MUET   = 0.55   # profil 10 au-dessus malgré 09 neutre

# Signal E — autorité disciplinaire
SEUIL_ANCRAGE_HAUT  = 0.80   # ancrage auditoire 10 au-dessus

# Signal F — cohérence axiologique
SEUIL_VALEURS_HAUT       = 0.90
SEUIL_RISQUE_BAS         = 0.10
SEUIL_PROFIL_AXIOLOGIQUE = 0.73

# Nombre minimum de signaux pour figurer dans la liste "haute priorité"
SEUIL_PRIORITE_HAUTE = 2

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

# =============================================================================
# CHARGEMENT
# =============================================================================

def trouver_json(json_dir: Path, pattern: str) -> Path:
    """Retourne le fichier JSON le plus récent correspondant au pattern."""
    fichiers = sorted(
        [f for f in json_dir.glob(pattern) if "progress" not in f.name],
        reverse=True,
    )
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier {pattern} dans {json_dir}.\n"
            f"Lancez d'abord le script correspondant."
        )
    return fichiers[0]


def charger_json(chemin: Path) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def aligner_sections(data09: dict, data10: dict) -> list[dict]:
    """
    Aligne les sections des deux JSON par position ordinale.

    Les deux scripts traitent le même corpus dans le même ordre, mais
    les identifiants peuvent différer (id vs None). On aligne par position.

    Retourne une liste de dicts fusionnés avec tous les scores des deux scripts.
    """
    secs09 = data09.get("sections", [])
    paras10 = data10.get("paragraphes", data10.get("sections", []))

    n = min(len(secs09), len(paras10))
    if n == 0:
        raise ValueError("Aucune section à aligner — vérifiez les fichiers JSON.")

    if len(secs09) != len(paras10):
        print(
            f"  ⚠ Nombre de sections différent : "
            f"09={len(secs09)}, 10={len(paras10)}. "
            f"Alignement sur les {n} premiers."
        )

    fused = []
    for i in range(n):
        s9 = secs09[i]
        s10 = paras10[i]
        fused.append({
            # Identifiant et titre depuis le 09 (le plus fiable)
            "idx"    : s9.get("id", i + 1),
            "titre"  : s9.get("titre", f"Paragraphe {i+1}"),
            # Scores 09
            "complet": s9.get("score_completude_toulmin", 0.5),
            "warrant": s9.get("score_coherence_warrant", 0.5),
            "probat" : s9.get("score_charge_probatoire", 0.5),
            "risque9": s9.get("score_risque_sophisme", 0.5),
            "robust" : s9.get("score_robustesse_globale", 0.5),
            "schema9": s9.get("schema_walton_dominant", "aucun"),
            "sophs"  : s9.get("sophismes_detectes", []),
            # Scores 10
            "force"  : s10.get("score_force_persuasive", 0.5),
            "ancrage": s10.get("score_ancrage_auditoire", 0.5),
            "valeurs": s10.get("score_coherence_valeurs", 0.5),
            "risque10": s10.get("score_risque_sophistique_rhethorique", 0.5),
            "profil" : s10.get("score_profil_argumentatif", 0.5),
            "tech10" : s10.get("technique_dominante", "indeterminee"),
            "aud"    : s10.get("type_auditoire", "indetermine"),
        })
    return fused


# =============================================================================
# CALCUL DES SIGNAUX
# =============================================================================

SIGNAL_DESC = {
    "A": ("rhéto_sans_preuve",
          "Robustesse logique faible mais profil rhétorique fort — "
          "l'auteur argumente efficacement sans étayer. "
          "Candidat prioritaire pour script A (Toulmin sur segment)."),
    "B": ("force_sans_grounds",
          "Force persuasive élevée mais charge probatoire très basse — "
          "rhétorique forte, fondement logique absent. "
          "Candidat pour scripts A + B combinés."),
    "C": ("double_sophisme",
          "Sophisme détecté par le 09 ET risque sophistique élevé en 10 — "
          "convergence de deux cadres sur une anomalie argumentative. "
          "Candidat urgent pour script A."),
    "D": ("09_muet_10_parle",
          "Le 09 n'a pas identifié d'argument (score neutre 0.50) "
          "mais le 10 voit une rhétorique active. "
          "Souvent : discours rapporté, conclusion narrative, liste biblio. "
          "À examiner pour confirmer si c'est voulu."),
    "E": ("autorite_disciplinaire",
          "Argument d'autorité disciplinaire (fort ancrage auditoire) "
          "sans grounds explicites. "
          "Fréquent et souvent non problématique — à surveiller dans les sections clés."),
    "F": ("coherence_axiologique",
          "Forte cohérence de valeurs, risque sophistique bas — "
          "zone rhétoriquement solide du texte. Signal positif."),
}


def calculer_signaux(p: dict, seuil_robust: float = SEUIL_ROBUST_BAS) -> list[str]:
    """Calcule les signaux actifs pour un paragraphe fusionné."""
    sigs = []

    # A — rhétorique sans preuve
    if p["robust"] < seuil_robust and p["profil"] >= SEUIL_PROFIL_HAUT:
        sigs.append("A")

    # B — force sans grounds
    if p["probat"] <= SEUIL_PROBAT_BAS and p["force"] >= SEUIL_FORCE_HAUT:
        sigs.append("B")

    # C — double sophisme
    if p["sophs"] and p["risque10"] >= SEUIL_RISQUE_SOPH:
        sigs.append("C")

    # D — 09 muet, 10 parle
    if (p["robust"] == 0.50 and p["warrant"] == 0.50
            and p["profil"] > SEUIL_PROFIL_MUET):
        sigs.append("D")

    # E — autorité disciplinaire sans grounds
    if (p["ancrage"] >= SEUIL_ANCRAGE_HAUT
            and p["probat"] <= SEUIL_PROBAT_BAS
            and "autorit" in p["tech10"]):
        sigs.append("E")

    # F — cohérence axiologique (signal positif)
    if (p["valeurs"] >= SEUIL_VALEURS_HAUT
            and p["risque10"] <= SEUIL_RISQUE_BAS
            and p["profil"] >= SEUIL_PROFIL_AXIOLOGIQUE):
        sigs.append("F")

    return sigs


def priorite(sigs: list[str]) -> str:
    """Détermine le niveau de priorité d'un paragraphe selon ses signaux."""
    critiques = [s for s in sigs if s in ("A", "B", "C")]
    if len(sigs) >= SEUIL_PRIORITE_HAUTE and critiques:
        return "haute"
    if critiques:
        return "moyenne"
    if sigs:
        return "info"
    return ""


# =============================================================================
# PORTRAIT RHÉTORIQUE AGRÉGÉ
# =============================================================================

def portrait_agregé(fused: list[dict]) -> dict:
    """
    Calcule les statistiques agrégées sur l'ensemble du corpus.

    Ces valeurs sont statistiquement plus robustes que les scores individuels :
    une mesure bruitée sur un paragraphe, multipliée par N, donne une
    signature d'ensemble qui dit quelque chose de réel sur le style du texte.
    """
    n = len(fused)
    if n == 0:
        return {}

    def moy(key):
        return round(sum(p[key] for p in fused) / n, 3)

    schemas = Counter(p["schema9"] for p in fused if p["schema9"] != "aucun")
    techs   = Counter(p["tech10"]  for p in fused
                      if p["tech10"] not in ("indeterminee",))
    auds    = Counter(p["aud"] for p in fused)

    return {
        "n"                   : n,
        "robust_moy"          : moy("robust"),
        "profil_moy"          : moy("profil"),
        "force_moy"           : moy("force"),
        "ancrage_moy"         : moy("ancrage"),
        "valeurs_moy"         : moy("valeurs"),
        "schema_dominant"     : schemas.most_common(1)[0] if schemas else ("—", 0),
        "schema_distribution" : schemas.most_common(),
        "tech_dominante"      : techs.most_common(1)[0] if techs else ("—", 0),
        "tech_distribution"   : techs.most_common(6),
        "aud_dominant"        : auds.most_common(1)[0] if auds else ("—", 0),
        "n_sophismes"         : sum(1 for p in fused if p["sophs"]),
    }


# =============================================================================
# RAPPORT MARKDOWN
# =============================================================================

def _barre(v: float, w: int = 20) -> str:
    r = round(v * w)
    return f"[{'█'*r}{'░'*(w-r)}] {v:.2f}"


def generer_rapport_md(
    fused: list[dict],
    portrait: dict,
    ts09: str,
    ts10: str,
    output_dir: Path,
    timestamp: str,
    seuil_robust: float = SEUIL_ROBUST_BAS,
) -> Path:
    """
    Génère zones_{timestamp}.md — rapport de navigation pour l'historien.

    Structure :
      1. Portrait rhétorique agrégé du texte entier
      2. Tableau des zones par niveau de priorité
      3. Détail des zones haute priorité avec explication des signaux
      4. Légende des six signaux
    """
    zones_avec_sigs = [(p, calculer_signaux(p, seuil_robust)) for p in fused]
    haute   = [(p, s) for p, s in zones_avec_sigs if priorite(s) == "haute"]
    moyenne = [(p, s) for p, s in zones_avec_sigs if priorite(s) == "moyenne"]
    info    = [(p, s) for p, s in zones_avec_sigs if priorite(s) == "info"]

    lignes = []

    # ── En-tête ───────────────────────────────────────────────────────────────
    lignes += [
        "# Zones prioritaires — navigation argumentative",
        "",
        f"**Timestamp** : {timestamp}  ",
        f"**Source 09** : `argumentation_{ts09}.json`  ",
        f"**Source 10** : `perelman_{ts10}.json`  ",
        f"**Paragraphes analysés** : {portrait['n']}  ",
        f"**Coût de ce script** : zéro (aucun appel LLM)  ",
        "",
        "---",
        "",
        "## ⚠ Comment utiliser ce rapport",
        "",
        "Ce rapport **désigne des zones à relire**, il ne juge pas des arguments.",
        "Les scores 09 et 10 au niveau du paragraphe sont des signaux d'orientation,",
        "non des métriques argumentatives fiables — voir l'avertissement épistémique",
        "du README.",
        "",
        "**Avant de lancer les scripts A, B ou C sur une zone désignée :**",
        "",
        "1. **Ne copiez pas le paragraphe seul.** Un paragraphe isolé est trop court",
        "   pour une analyse Toulmin ou RST fiable, et le LLM peut confondre le",
        "   discours de l'auteur et un argumentaire rapporté (citation, résumé d'une",
        "   position adverse, discours d'époque).",
        "   Identifiez l'**unité argumentative** dont fait partie ce paragraphe :",
        "   les paragraphes adjacents qui forment avec lui un développement cohérent.",
        "   C'est ce segment qu'il faut fournir aux scripts.",
        "",
        "2. **Deux façons de fournir le segment :**",
        "   - Copier-coller dans la console (terminer par `###FIN###`)",
        "   - Fichier `.txt` : `python A_toulmin_segment.py --fichier mon_segment.txt`",
        "   La deuxième option est recommandée pour les segments longs ou si vous",
        "   relancez plusieurs fois la même analyse.",
        "",
        "3. **Titre précis obligatoire.** Si le passage rapporte un argumentaire",
        "   qui n'est pas le vôtre, dites-le dans le titre :",
        "   *'Stratégie rhétorique de Hugo dans la pétition de 1880'*",
        "   plutôt que *'Paragraphe 77'*. Le script A identifie le discours rapporté",
        "   avant d'appliquer Toulmin, mais seulement si le titre le signale.",
        "",
        "**Quel script choisir ?**",
        "",
        "| Script | Question posée | Quand l'utiliser |",
        "|---|---|---|",
        "| **A** — Toulmin / Adam / Walton | Quelle est la structure logique de cet argument ? | Signaux A, B, C — problème de solidité argumentative |",
        "| **B** — Perelman | Quelles sont les stratégies rhétoriques ? Comment l'auditoire est-il construit ? | Signaux A, E, F — problème ou confirmation rhétorique |",
        "| **C** — RST | Comment les phrases s'articulent-elles ? La cohérence discursive est-elle solide ? | Signal D (09 muet), passages où quelque chose ne s'enchaîne pas bien malgré un profil rhétorique correct |",
        "| **D** — Synthèse | Que disent ensemble A, B et C ? | Après avoir lancé au moins deux des trois scripts sur le même segment |",
        "",
        "> A et B analysent *ce que* le texte argue et *comment* il le fait.",
        "> C analyse *comment les pièces s'assemblent* — c'est l'outil pour les",
        "> passages où l'intuition dit 'quelque chose ne s'enchaîne pas' sans",
        "> qu'on sache exactement où.",
        "",
        "---",
        "",
    ]

    # ── Portrait agrégé ───────────────────────────────────────────────────────
    lignes += [
        "## Portrait rhétorique du texte",
        "",
        "> *Ces valeurs agrégées sont statistiquement plus robustes*  ",
        "> *que les scores individuels par paragraphe.*",
        "",
        "| Dimension | Score moyen |",
        "|---|---|",
        f"| Robustesse logique (09) | `{_barre(portrait['robust_moy'])}` |",
        f"| Profil rhétorique (10)  | `{_barre(portrait['profil_moy'])}` |",
        f"| Force persuasive (10)   | `{_barre(portrait['force_moy'])}` |",
        f"| Ancrage auditoire (10)  | `{_barre(portrait['ancrage_moy'])}` |",
        f"| Cohérence valeurs (10)  | `{_barre(portrait['valeurs_moy'])}` |",
        "",
        f"**Schème Walton dominant** : `{portrait['schema_dominant'][0]}` "
        f"({portrait['schema_dominant'][1]} occurrences)",
        "",
        "Distribution des schèmes Walton (09) :",
        "",
    ]
    for k, v in portrait["schema_distribution"]:
        lignes.append(f"- `{k}` : {v} paragraphes ({v/portrait['n']*100:.0f}%)")
    lignes.append("")

    lignes += [
        f"**Technique Perelman dominante** : `{portrait['tech_dominante'][0]}` "
        f"({portrait['tech_dominante'][1]} occurrences)",
        "",
        "Distribution des techniques Perelman (10) :",
        "",
    ]
    for k, v in portrait["tech_distribution"]:
        lignes.append(f"- `{k}` : {v} paragraphes")
    lignes += [
        "",
        f"**Auditoire dominant** : `{portrait['aud_dominant'][0]}`  ",
        f"**Sections avec sophismes** : {portrait['n_sophismes']} "
        f"({portrait['n_sophismes']/portrait['n']*100:.0f}%)",
        "",
        "---",
        "",
    ]

    # ── Résumé des zones ───────────────────────────────────────────────────────
    lignes += [
        "## Zones à explorer",
        "",
        f"| Priorité | Nombre de zones | Action recommandée |",
        f"|---|---|---|",
        f"| 🔴 Haute   | {len(haute)} | Scripts A + B (ou A seul selon le signal) |",
        f"| 🟡 Moyenne | {len(moyenne)} | Script A en priorité |",
        f"| 🔵 Info    | {len(info)} | Signaux positifs — zones rhétoriquement solides |",
        "",
        "---",
        "",
    ]

    # ── Zones haute priorité (détaillées) ─────────────────────────────────────
    if haute:
        lignes += ["## 🔴 Zones haute priorité\n"]
        for p, sigs in haute:
            sigs_str = " + ".join(
                f"`{s}:{SIGNAL_DESC[s][0]}`" for s in sigs
            )
            lignes += [
                f"### § {p['idx']} — {p['titre'][:60]}",
                "",
                f"**Signaux** : {sigs_str}",
                "",
                f"| Score | 09 | 10 |",
                f"|---|---|---|",
                f"| Robustesse / Profil | `{p['robust']:.2f}` | `{p['profil']:.2f}` |",
                f"| Warrant / Force     | `{p['warrant']:.2f}` | `{p['force']:.2f}` |",
                f"| Probatoire / Ancrage| `{p['probat']:.2f}` | `{p['ancrage']:.2f}` |",
                f"| Risque sophisme     | `{p['risque9']:.2f}` | `{p['risque10']:.2f}` |",
                "",
                f"**Schème Walton** : `{p['schema9']}` &nbsp;|&nbsp; "
                f"**Technique Perelman** : `{p['tech10']}` &nbsp;|&nbsp; "
                f"**Auditoire** : `{p['aud']}`",
                "",
            ]
            if p["sophs"]:
                lignes.append(f"⚑ **Sophismes détectés** : {', '.join(p['sophs'])}")
                lignes.append("")
            # Explication des signaux actifs
            for s in sigs:
                lignes.append(f"> **{s} — {SIGNAL_DESC[s][0]}** : {SIGNAL_DESC[s][1]}")
            # Recommandation de script selon les signaux actifs
            scripts_recommandes = []
            if any(s in sigs for s in ("A", "B", "C")):
                scripts_recommandes.append(
                    "**Script A** (Toulmin / Adam / Walton) — "
                    "signal de fragilité argumentative détecté."
                )
            if any(s in sigs for s in ("A", "E", "F")):
                scripts_recommandes.append(
                    "**Script B** (Perelman) — "
                    "signal rhétorique : autorité, cohérence ou force persuasive."
                )
            if "D" in sigs:
                scripts_recommandes.append(
                    "**Script C** (RST) — signal D : le 09 n'a pas identifié d'argument "
                    "mais le 10 voit une rhétorique active. Le problème est peut-être "
                    "de cohérence discursive entre les phrases plutôt qu'argumentatif. "
                    "C est l'outil pour les passages où quelque chose ne s'enchaîne "
                    "pas bien sans qu'on sache exactement où."
                )
            if not scripts_recommandes:
                scripts_recommandes.append(
                    "**Script A** en priorité pour confirmer le signal."
                )

            lignes += ["", "**Scripts recommandés pour cette zone :**", ""]
            for rec in scripts_recommandes:
                lignes.append(f"- {rec}")
            lignes += [
                "",
                "**Avant de lancer :** délimitez l'unité argumentative dont fait partie"
                " ce paragraphe (les paragraphes adjacents qui forment avec lui un"
                " développement cohérent). Fournissez ce segment, pas le paragraphe seul.",
                "Fichier `.txt` recommandé :"
                " `python A_toulmin_segment.py --fichier mon_segment.txt`",
                "ou copier-coller dans la console (terminer par `###FIN###`).",
                "Titre précis obligatoire — si le passage rapporte un argumentaire"
                " qui n'est pas le vôtre, signalez-le dans le titre.",
                "",
            ]
            lignes += ["---", ""]

    # ── Zones moyenne priorité (tableau) ──────────────────────────────────────
    if moyenne:
        lignes += [
            "## 🟡 Zones moyenne priorité",
            "",
            "| § | Titre | Rob. | Profil | Signal |",
            "|---|---|---|---|---|",
        ]
        for p, sigs in moyenne:
            sigs_str = " ".join(f"`{s}`" for s in sigs)
            lignes.append(
                f"| {p['idx']} | {p['titre'][:45]} "
                f"| {p['robust']:.2f} | {p['profil']:.2f} | {sigs_str} |"
            )
        lignes += ["", "---", ""]

    # ── Zones info ────────────────────────────────────────────────────────────
    if info:
        lignes += [
            "## 🔵 Zones à forte cohérence (signal F)",
            "",
            f"Ces {len(info)} paragraphes présentent une forte cohérence "
            "axiologique et un risque sophistique bas — zones rhétoriquement "
            "solides du texte. Pas d'action requise sauf si une révision "
            "ultérieure les modifie.",
            "",
            "| § | Titre | Profil | Valeurs | Technique |",
            "|---|---|---|---|---|",
        ]
        for p, _ in info:
            lignes.append(
                f"| {p['idx']} | {p['titre'][:45]} "
                f"| {p['profil']:.2f} | {p['valeurs']:.2f} | `{p['tech10']}` |"
            )
        lignes += ["", "---", ""]

    # ── Légende ───────────────────────────────────────────────────────────────
    lignes += [
        "## Légende des signaux",
        "",
        "| Signal | Nom | Interprétation |",
        "|---|---|---|",
    ]
    for k, (nom, desc) in SIGNAL_DESC.items():
        lignes.append(f"| `{k}` | {nom} | {desc[:80]}… |")

    lignes += [
        "",
        "---",
        "",
        "*Rapport généré par `11_zones_prioritaires.py`.*  ",
        "*Aucun appel LLM — coût zéro.*  ",
        f"*Seuils modifiables en tête du script.*",
    ]

    contenu   = "\n".join(lignes)
    chemin_md = output_dir / f"zones_{timestamp}.md"
    chemin_md.write_text(contenu, encoding="utf-8")
    return chemin_md


# =============================================================================
# AFFICHAGE CONSOLE
# =============================================================================

def afficher_console(fused: list[dict], portrait: dict, seuil_robust: float = SEUIL_ROBUST_BAS) -> None:
    """Résumé affiché pendant le run."""
    zones_avec_sigs = [(p, calculer_signaux(p, seuil_robust)) for p in fused]
    haute   = [(p, s) for p, s in zones_avec_sigs if priorite(s) == "haute"]
    moyenne = [(p, s) for p, s in zones_avec_sigs if priorite(s) == "moyenne"]
    info    = [(p, s) for p, s in zones_avec_sigs if priorite(s) == "info"]

    print(f"\n{'═'*70}")
    print(f"  PORTRAIT RHÉTORIQUE — {portrait['n']} paragraphes")
    print(f"{'═'*70}")
    print(f"  Robustesse logique (09) moy. : {portrait['robust_moy']:.3f}")
    print(f"  Profil rhétorique  (10) moy. : {portrait['profil_moy']:.3f}")
    print(f"  Schème Walton dominant        : {portrait['schema_dominant'][0]} "
          f"({portrait['schema_dominant'][1]} occ.)")
    print(f"  Technique Perelman dominante  : {portrait['tech_dominante'][0]} "
          f"({portrait['tech_dominante'][1]} occ.)")
    print(f"  Sections avec sophismes       : {portrait['n_sophismes']}")

    print(f"\n{'─'*70}")
    print(f"  ZONES À EXPLORER")
    print(f"{'─'*70}")
    print(f"  🔴 Haute priorité  : {len(haute):3d} zones")
    print(f"  🟡 Moyenne         : {len(moyenne):3d} zones")
    print(f"  🔵 Info (signal F) : {len(info):3d} zones")

    if haute:
        print(f"\n  Zones haute priorité :")
        for p, sigs in haute[:10]:
            sigs_str = "+".join(sigs)
            print(f"    §{p['idx']:3d}  rob={p['robust']:.2f} prof={p['profil']:.2f}"
                  f"  [{sigs_str}]  {p['titre'][:40]}")
        if len(haute) > 10:
            print(f"    … et {len(haute)-10} autres (voir le rapport MD)")

    print(f"\n{'─'*70}")
    print(f"  ⚠  AVANT DE LANCER LES SCRIPTS A, B OU C")
    print(f"{'─'*70}")
    print(f"  Ces scripts attendent un SEGMENT cohérent, pas un paragraphe isolé.")
    print(f"  Un paragraphe seul est trop court pour une analyse Toulmin fiable")
    print(f"  et le LLM peut confondre le discours de l'auteur et le discours")
    print(f"  rapporté (ex : l'historien qui résume une position adverse).")
    print()
    print(f"  Pour chaque zone désignée ci-dessus :")
    print(f"    1. Identifiez les paragraphes adjacents qui forment avec elle")
    print(f"       une unité argumentative cohérente (section, développement).")
    print(f"       C'est CE SEGMENT qu'il faut fournir aux scripts A/B/C,")
    print(f"       pas le paragraphe seul.")
    print()
    print(f"    2. Si le passage rapporte un argument d'époque, une citation")
    print(f"       ou une position adverse, indiquez-le dans le titre du segment")
    print(f"       (ex : 'Analyse de la stratégie rhétorique de Hugo, 1880')")
    print(f"       pour que le LLM sache ce qui est discours de l'auteur")
    print(f"       et ce qui est discours rapporté.")
    print(f"{'═'*70}\n")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="11_zones_prioritaires — cartographie des zones à explorer "
                    "(croisement 09 × 10, aucun appel LLM)."
    )
    parser.add_argument("--arg09", type=str, default=None,
                        help="Fichier argumentation_*.json (script 09). "
                             "Si absent : prend le plus récent dans JSON_DIR.")
    parser.add_argument("--per10", type=str, default=None,
                        help="Fichier perelman_*.json (script 10). "
                             "Si absent : prend le plus récent dans JSON_DIR.")
    parser.add_argument("--seuil_robust", type=float, default=SEUIL_ROBUST_BAS,
                        help=f"Override SEUIL_ROBUST_BAS (défaut : {SEUIL_ROBUST_BAS}).")
    args = parser.parse_args()

    seuil_robust = args.seuil_robust
    if seuil_robust != SEUIL_ROBUST_BAS:
        print(f"  Seuil robustesse overridé : {seuil_robust}")

    json_dir   = Path(JSON_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Chargement des JSON ───────────────────────────────────────────────────
    if args.arg09:
        chemin09 = Path(args.arg09)
        if not chemin09.is_absolute():
            chemin09 = json_dir / chemin09
    else:
        print("Recherche du JSON 09 le plus récent…")
        chemin09 = trouver_json(json_dir, "argumentation_*.json")

    if args.per10:
        chemin10 = Path(args.per10)
        if not chemin10.is_absolute():
            chemin10 = json_dir / chemin10
    else:
        print("Recherche du JSON 10 le plus récent…")
        chemin10 = trouver_json(json_dir, "perelman_*.json")

    print(f"  09 : {chemin09.name}")
    print(f"  10 : {chemin10.name}")

    data09 = charger_json(chemin09)
    data10 = charger_json(chemin10)

    ts09 = data09.get("timestamp", chemin09.stem.replace("argumentation_", ""))
    ts10 = data10.get("timestamp", chemin10.stem.replace("perelman_", ""))

    # ── Alignement et calcul ──────────────────────────────────────────────────
    print("\nAlignement des sections…")
    fused = aligner_sections(data09, data10)
    print(f"  {len(fused)} sections alignées.")

    portrait = portrait_agregé(fused)

    # ── Affichage console ─────────────────────────────────────────────────────
    afficher_console(fused, portrait, seuil_robust)

    # ── Rapport MD ────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chemin_md = generer_rapport_md(
        fused, portrait, ts09, ts10, output_dir, timestamp, seuil_robust
    )
    print(f"✅ Rapport MD : {chemin_md}")
    print(f"\n   Étape suivante : ouvrez {chemin_md.name} et lancez")
    print(f"   les scripts A, B ou C sur les zones haute priorité.\n")


if __name__ == "__main__":
    main()
