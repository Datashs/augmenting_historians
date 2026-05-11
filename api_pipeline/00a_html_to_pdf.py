#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00a_html_to_pdf.py
==================
Conversion des snapshots HTML du stockage Zotero en PDF.

Rôle : Parcourt le CSV exporté depuis Zotero, identifie les items dont
le seul fichier attaché est un HTML (snapshot de page web sauvegardé par
le connecteur Zotero), et les convertit en PDF dans le même dossier
storage de Zotero. Le CSV Zotero n'est PAS modifié — la mise à jour
des pièces jointes dans Zotero se fait manuellement après vérification
(voir section APRÈS L'EXÉCUTION ci-dessous).

Ce script est la première étape de la voie A (intégration Zotero).
Il doit être lancé AVANT 00_zotero_import.py.

POURQUOI CONVERTIR LES HTML EN PDF ?
─────────────────────────────────────
Le pipeline d'extraction de texte (01_extract_text.py) travaille
uniquement sur des PDF. Les snapshots HTML sauvegardés par Zotero sont
des pages web complètes avec ressources (images, CSS) — ils contiennent
souvent du texte académique pertinent (articles en accès libre, billets
de blog scientifiques, pages institutionnelles) qui serait perdu si on
les ignorait.

MÉTHODE DE CONVERSION
──────────────────────
Ce script utilise WeasyPrint, une librairie Python pure qui convertit
HTML + CSS en PDF. Avantages : pas de dépendance externe, gère bien
les snapshots Zotero qui incluent des ressources locales (images, CSS
sauvegardés dans des sous-dossiers). Limite : les pages très complexes
(JavaScript rendu côté client) peuvent produire des PDFs incomplets —
dans ce cas, la conversion via le navigateur en headless est préférable
(voir paramètre FALLBACK_BROWSER ci-dessous).

Installation :
    pip install weasyprint
    # Sur macOS, peut nécessiter :
    brew install pango

APRÈS L'EXÉCUTION
──────────────────
Les PDFs sont créés dans les dossiers storage Zotero à côté des HTML.
Pour que Zotero les reconnaisse comme pièces jointes :
    1. Dans Zotero, clic droit sur l'item → "Ajouter une pièce jointe"
       → "Attacher un fichier stocké copié"
    2. Naviguer vers le PDF créé dans storage/XXXXXXXX/
Ou laisser le script 00_zotero_import.py les détecter automatiquement
(il cherche tous les PDF dans le dossier storage de chaque item,
pas seulement ceux listés dans le CSV).

UTILISATION
───────────
    python 00a_html_to_pdf.py
    python 00a_html_to_pdf.py --csv ma_bibliotheque.csv
    python 00a_html_to_pdf.py --dry-run   # affiche ce qui serait converti
    python 00a_html_to_pdf.py --force     # reconvertit même si PDF existe déjà
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

# Chemin vers le CSV exporté depuis Zotero
# Exporter via : Fichier → Exporter la bibliothèque → Format CSV
CSV_ZOTERO = "IAHistoire.csv"

# Dossier racine du stockage Zotero
# Par défaut sur macOS : ~/Zotero/storage
# Modifier si votre stockage Zotero est ailleurs
import os
ZOTERO_STORAGE = os.path.expanduser("~/Zotero/storage")

# Convertir même si un PDF existe déjà pour cet item ?
# False = ignorer les items qui ont déjà un PDF (recommandé)
# True  = forcer la reconversion (utile si le PDF existant est corrompu)
FORCE_RECONVERSION = False

# Largeur de page PDF en millimètres (A4 = 210)
PAGE_WIDTH_MM = 210

# Marges en millimètres
MARGIN_MM = 20

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import sys
from pathlib import Path
from datetime import datetime

# =============================================================================
# FONCTIONS — CHARGEMENT DU CSV
# =============================================================================

def charger_items_html(chemin_csv: Path) -> list[dict]:
    """
    Lit le CSV Zotero et retourne les items dont au moins un fichier
    attaché est un HTML et qui n'ont pas encore de PDF.

    Si FORCE_RECONVERSION est True, retourne tous les items avec HTML,
    qu'ils aient un PDF ou non.

    Returns:
        Liste de dicts avec les clés :
            auteur, titre, annee, type_item,
            chemin_html (Path), chemin_pdf_cible (Path),
            a_deja_pdf (bool)
    """
    if not chemin_csv.exists():
        raise FileNotFoundError(
            f"CSV Zotero introuvable : {chemin_csv.resolve()}\n"
            "Exporter depuis Zotero : Fichier → Exporter la bibliothèque → CSV"
        )

    items = []
    with open(chemin_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            att = row.get("File Attachments", "").strip()
            if not att:
                continue

            chemins = [c.strip() for c in att.split(";") if c.strip()]
            htmls = [Path(c) for c in chemins if c.lower().endswith(".html")]
            pdfs  = [Path(c) for c in chemins if c.lower().endswith(".pdf")]

            if not htmls:
                continue

            html_path = htmls[0]   # on prend le premier HTML

            # Cible PDF : même dossier, même nom, extension .pdf
            pdf_cible = html_path.with_suffix(".pdf")

            a_deja_pdf = bool(pdfs) or pdf_cible.exists()

            items.append({
                "auteur"      : row.get("Author", ""),
                "titre"       : row.get("Title", ""),
                "annee"       : row.get("Publication Year", ""),
                "type_item"   : row.get("Item Type", ""),
                "chemin_html" : html_path,
                "chemin_pdf"  : pdf_cible,
                "a_deja_pdf"  : a_deja_pdf,
            })

    return items


# =============================================================================
# FONCTIONS — CONVERSION
# =============================================================================

def convertir_avec_weasyprint(chemin_html: Path, chemin_pdf: Path) -> bool:
    """
    Convertit un fichier HTML en PDF via WeasyPrint.

    WeasyPrint résout automatiquement les ressources locales (images, CSS)
    si elles sont dans le même dossier ou un sous-dossier — ce qui est
    le cas pour les snapshots Zotero.

    Returns:
        True si la conversion a réussi, False sinon.
    """
    try:
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        print("❌ WeasyPrint non installé. Lancer : pip install weasyprint")
        sys.exit(1)

    try:
        font_config = FontConfiguration()
        css_page = CSS(
            string=f"""
            @page {{
                size: {PAGE_WIDTH_MM}mm auto;
                margin: {MARGIN_MM}mm;
            }}
            body {{
                font-family: serif;
                font-size: 11pt;
                line-height: 1.5;
            }}
            """,
            font_config=font_config,
        )
        HTML(filename=str(chemin_html)).write_pdf(
            str(chemin_pdf),
            stylesheets=[css_page],
            font_config=font_config,
        )
        return True
    except Exception as e:
        print(f"    ⚠ Erreur WeasyPrint : {e}")
        return False


# =============================================================================
# RAPPORT
# =============================================================================

def generer_rapport(
    convertis: list[dict],
    ignores: list[dict],
    echecs: list[dict],
    timestamp: str,
) -> Path:
    """
    Génère un rapport Markdown de la conversion.
    Utile pour vérifier manuellement les items convertis avant de
    mettre à jour Zotero.
    """
    lignes = [
        "# Rapport de conversion HTML → PDF",
        "",
        f"**Timestamp** : {timestamp}  ",
        f"**Convertis avec succès** : {len(convertis)}  ",
        f"**Ignorés (PDF déjà présent)** : {len(ignores)}  ",
        f"**Échecs** : {len(echecs)}  ",
        "",
        "---",
        "",
    ]

    if convertis:
        lignes += ["## Convertis avec succès", ""]
        for item in convertis:
            auteur = item['auteur'].split(';')[0].strip()[:40]
            lignes.append(
                f"- {auteur} ({item['annee']}) — *{item['titre'][:60]}*"
            )
            lignes.append(f"  `{item['chemin_pdf'].name}`")
        lignes.append("")

    if echecs:
        lignes += ["## Échecs de conversion", ""]
        for item in echecs:
            auteur = item['auteur'].split(';')[0].strip()[:40]
            lignes.append(
                f"- ❌ {auteur} ({item['annee']}) — *{item['titre'][:60]}*"
            )
            lignes.append(
                f"  HTML : `{item['chemin_html']}`"
            )
        lignes += [
            "",
            "> Pour les échecs, ouvrez le HTML dans un navigateur et",
            "> exportez manuellement en PDF (Fichier → Imprimer → Enregistrer en PDF).",
            "",
        ]

    if ignores:
        lignes += [
            "## Ignorés (PDF déjà présent)",
            "",
            f"Ces {len(ignores)} items ont déjà un PDF attaché — non reconvertis.",
            "Relancer avec `--force` pour les reconvertir.",
            "",
        ]

    lignes += [
        "---",
        "",
        "## Étape suivante",
        "",
        "Lancer `00_zotero_import.py` pour construire `metadata_refs.json`.",
        "Ce script détectera automatiquement les nouveaux PDFs créés ici.",
        "",
        "*Rapport généré par `00a_html_to_pdf.py`.*",
    ]

    chemin_rapport = Path(f"rapport_html_pdf_{timestamp}.md")
    chemin_rapport.write_text("\n".join(lignes), encoding="utf-8")
    return chemin_rapport


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="00a_html_to_pdf — Convertit les snapshots HTML Zotero en PDF."
    )
    parser.add_argument("--csv", type=str, default=CSV_ZOTERO,
                        help=f"CSV Zotero (défaut : {CSV_ZOTERO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche ce qui serait converti sans rien faire.")
    parser.add_argument("--force", action="store_true",
                        help="Reconvertit même si un PDF existe déjà.")
    args = parser.parse_args()

    chemin_csv = Path(args.csv)
    force      = args.force or FORCE_RECONVERSION

    print("=" * 60)
    print("  CONVERSION HTML → PDF (snapshots Zotero)")
    print("=" * 60)
    print(f"  CSV      : {chemin_csv}")
    print(f"  Storage  : {ZOTERO_STORAGE}")
    print(f"  Force    : {force}")
    print()

    # Chargement
    try:
        items = charger_items_html(chemin_csv)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"  Items avec HTML trouvés : {len(items)}")

    # Filtrage
    a_convertir = []
    ignores     = []
    for item in items:
        if item["a_deja_pdf"] and not force:
            ignores.append(item)
        else:
            a_convertir.append(item)

    print(f"  À convertir             : {len(a_convertir)}")
    print(f"  Ignorés (PDF existant)  : {len(ignores)}")
    print()

    if not a_convertir:
        print("  Rien à convertir.")
        sys.exit(0)

    if args.dry_run:
        print("  Mode --dry-run : aucune conversion effectuée.\n")
        print("  Seraient convertis :")
        for item in a_convertir:
            auteur = item['auteur'].split(';')[0].strip()[:40]
            print(f"    {auteur[:30]:<30} | {item['titre'][:50]}")
            print(f"    HTML : {str(item['chemin_html'])[-70:]}")
            print(f"    PDF  : {str(item['chemin_pdf'])[-70:]}")
            print()
        sys.exit(0)

    # Conversion
    convertis = []
    echecs    = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, item in enumerate(a_convertir, 1):
        auteur = item['auteur'].split(';')[0].strip()[:35]
        print(f"  [{i:2d}/{len(a_convertir)}] {auteur} ({item['annee']}) — "
              f"{item['titre'][:45]}")

        html_path = item['chemin_html']
        pdf_path  = item['chemin_pdf']

        if not html_path.exists():
            print(f"    ⚠ HTML introuvable : {html_path}")
            echecs.append(item)
            continue

        ok = convertir_avec_weasyprint(html_path, pdf_path)
        if ok:
            taille = pdf_path.stat().st_size // 1024
            print(f"    ✅ {pdf_path.name} ({taille} Ko)")
            item['chemin_pdf'] = pdf_path
            convertis.append(item)
        else:
            echecs.append(item)

    # Rapport
    print()
    print("=" * 60)
    print(f"  Convertis : {len(convertis)}")
    print(f"  Échecs    : {len(echecs)}")
    print("=" * 60)

    rapport = generer_rapport(convertis, ignores, echecs, timestamp)
    print(f"\n  Rapport   : {rapport}")
    print()

    if echecs:
        print("  ⚠ Pour les échecs : ouvrir le HTML dans un navigateur")
        print("    et exporter en PDF via Fichier → Imprimer → Enregistrer en PDF.")
        print()

    if convertis:
        print("  Étape suivante : python 00_zotero_import.py")
        print("  (détectera automatiquement les nouveaux PDFs)\n")


if __name__ == "__main__":
    main()
