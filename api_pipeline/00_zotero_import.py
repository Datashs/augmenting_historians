#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_zotero_import.py
===================
Voie A — Import initial depuis une bibliothèque Zotero.

Rôle : Lit le CSV exporté depuis Zotero, construit metadata_refs.json
(dictionnaire {nom_fichier → référence bibliographique formée}), et
copie les PDFs dans le dossier pdfs/ du projet.

POSITION DANS LE PIPELINE (voie A)
────────────────────────────────────
  00a_html_to_pdf.py     → PDFs créés dans storage Zotero
  00_zotero_import.py    → pdfs/ + metadata_refs.json      (ce script)
  01_extract_text.py     → extracted_text/corpus.txt
  02_chunk_corpus.py     → extracted_text/chunks.json
  03_build_embeddings.py → vector_store/

  Mise à jour ultérieure :
  00b_zotero_update.py   → ajoute les nouveautés sans reconstruire

CE QUE PRODUIT CE SCRIPT
─────────────────────────
  pdfs/Noiriel1988.pdf, pdfs/Schor1996.pdf, ...

  metadata_refs.json :
    {
      "Noiriel1988.pdf": {
        "key"       : "AB12CD34",
        "auteur"    : "Noiriel, Gérard",
        "titre"     : "Le Creuset français",
        "annee"     : "1988",
        "type_item" : "book",
        "ref_courte": "Noiriel (1988)",
        "ref_longue": "Noiriel G., Le Creuset français, Seuil, 1988"
      }, ...
    }

FORMAT DES RÉFÉRENCES
──────────────────────
  Référence courte (tableaux MD) :
    1 auteur   → Noiriel (1988)
    2 auteurs  → Noiriel & Schor (1996)
    3+ auteurs → Noiriel et al. (2000)

  Référence longue (sections détaillées) :
    Article  : Auteur, "Titre", Revue, Vol(N), Année, pp. X-Y
    Livre    : Auteur, Titre, Éditeur, Lieu, Année
    Chapitre : Auteur, "Titre", in Dir. (dir.), Titre livre, Éd., Année

GESTION DES CAS PARTICULIERS
──────────────────────────────
  - Item sans PDF : ignoré, noté dans le rapport
  - Item sans année : référence sans date
  - Nom de fichier doublon : suffixe _a, _b... ajouté
  - PDF introuvable sur disque : ignoré avec avertissement

UTILISATION
───────────
  python 00_zotero_import.py
  python 00_zotero_import.py --csv MaBiblio.csv
  python 00_zotero_import.py --dry-run
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

import os

CSV_ZOTERO     = "IAHistoire.csv"
ZOTERO_STORAGE = os.path.expanduser("~/Zotero/storage")
PDF_DIR        = "pdfs"
METADATA_REFS  = "metadata_refs.json"

# "copie" : copie physique (recommandé)
# "lien"  : lien symbolique (économise l'espace)
MODE_FICHIER = "copie"

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# PARSING DES AUTEURS
# =============================================================================

def parser_auteurs(champ: str) -> list[dict]:
    """
    Parse le champ Author du CSV Zotero.
    Format : "Nom, Prénom; Nom2, Prénom2"
    """
    if not champ.strip():
        return []
    auteurs = []
    for a in champ.split(";"):
        a = a.strip()
        if not a:
            continue
        if "," in a:
            parties = a.split(",", 1)
            nom    = parties[0].strip()
            prenom = parties[1].strip()
        else:
            nom, prenom = a, ""
        initiale = prenom[0] + "." if prenom else ""
        auteurs.append({"nom": nom, "prenom": prenom, "initiale": initiale})
    return auteurs


def auteurs_court(auteurs: list[dict]) -> str:
    if not auteurs:
        return "Anonyme"
    if len(auteurs) == 1:
        return auteurs[0]["nom"]
    if len(auteurs) == 2:
        return f"{auteurs[0]['nom']} & {auteurs[1]['nom']}"
    return f"{auteurs[0]['nom']} et al."


def auteurs_long(auteurs: list[dict]) -> str:
    if not auteurs:
        return "Anonyme"
    parties = []
    for a in auteurs:
        parties.append(f"{a['nom']} {a['initiale']}".strip())
    if len(parties) == 1:
        return parties[0]
    if len(parties) == 2:
        return f"{parties[0]} & {parties[1]}"
    return ", ".join(parties[:-1]) + f" & {parties[-1]}"


# =============================================================================
# CONSTRUCTION DES RÉFÉRENCES
# =============================================================================

def ref_courte(auteurs: list[dict], annee: str) -> str:
    noms = auteurs_court(auteurs)
    return f"{noms} ({annee})" if annee else noms


def ref_longue(row: dict, auteurs: list[dict]) -> str:
    noms  = auteurs_long(auteurs)
    titre = row.get("Title", "").strip()
    annee = row.get("Publication Year", "").strip()
    revue = row.get("Publication Title", "").strip()
    vol   = row.get("Volume", "").strip()
    num   = row.get("Issue", "").strip()
    pages = row.get("Pages", "").strip()
    edit  = row.get("Publisher", "").strip()
    lieu  = row.get("Place", "").strip()
    type_ = row.get("Item Type", "").strip()
    eds   = row.get("Editor", "").strip()

    vol_num = f"{vol}({num})" if vol and num else vol
    pp      = f"pp. {pages}" if pages else ""

    def join(*parties):
        return ", ".join(p for p in parties if p)

    if type_ == "journalArticle":
        return join(noms, f'"{titre}"', revue, vol_num, annee, pp)

    elif type_ == "book":
        return join(noms, titre, edit, lieu, annee)

    elif type_ == "bookSection":
        dirs = ""
        if eds:
            dirs = auteurs_long(parser_auteurs(eds)) + " (dir.)"
        in_livre = f"in {dirs}, {revue}" if dirs and revue else (
                   f"in {revue}" if revue else "")
        return join(noms, f'"{titre}"', in_livre, edit, annee, pp)

    elif type_ == "conferencePaper":
        return join(noms, f'"{titre}"', revue, annee, pp)

    elif type_ in ("preprint", "report"):
        label = "preprint" if type_ == "preprint" else "rapport"
        return join(noms, titre, f"[{label}]", annee)

    else:
        return join(noms, titre, annee)


# =============================================================================
# NOM DE FICHIER
# =============================================================================

def nom_fichier(auteurs: list[dict], annee: str, existants: set) -> str:
    """Construit un nom de fichier propre et unique : Auteur{Année}.pdf"""
    import unicodedata

    def norm(s: str) -> str:
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return re.sub(r"[^a-zA-Z0-9]", "", s)

    base = "".join(norm(a["nom"]) for a in auteurs[:2])
    base += annee if annee else "SansDate"

    candidat = base + ".pdf"
    if candidat not in existants:
        return candidat
    for lettre in "abcdefghijklmnopqrstuvwxyz":
        candidat = base + lettre + ".pdf"
        if candidat not in existants:
            return candidat
    return base + str(len(existants)) + ".pdf"


# =============================================================================
# RÉSOLUTION DU PDF
# =============================================================================

def cle_storage(att: str) -> str | None:
    """Extrait la clé du dossier storage depuis un chemin Zotero."""
    m = re.search(r"storage[/\\]([A-Z0-9]{8})[/\\]", att, re.IGNORECASE)
    return m.group(1) if m else None


def trouver_pdf(att: str, zotero_storage: Path) -> Path | None:
    """
    Résout le chemin vers le PDF d'un item.
    Cherche d'abord dans le CSV, puis dans le dossier storage.
    """
    if att:
        chemins = [c.strip() for c in att.split(";") if c.strip()]
        pdfs = [Path(c) for c in chemins if c.lower().endswith(".pdf")]
        for p in pdfs:
            if p.exists():
                return p

        # Fallback : chercher dans le dossier storage
        cle = cle_storage(att)
        if cle:
            dossier = zotero_storage / cle
            if dossier.exists():
                pdfs_dossier = list(dossier.glob("*.pdf"))
                if pdfs_dossier:
                    return pdfs_dossier[0]

    return None


# =============================================================================
# TRAITEMENT PRINCIPAL
# =============================================================================

def traiter_csv(
    chemin_csv: Path,
    zotero_storage: Path,
    pdf_dir: Path,
    mode: str,
    dry_run: bool,
) -> tuple[dict, list, list, list]:
    """
    Lit le CSV et produit métadonnées + PDFs.

    Returns:
        metadata, importes, sans_pdf, erreurs
    """
    metadata: dict  = {}
    importes: list  = []
    sans_pdf: list  = []
    erreurs: list   = []
    noms_existants: set = set()

    with open(chemin_csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    print(f"  {len(rows)} items dans le CSV.")

    for row in rows:
        key     = row.get("Key", "").strip()
        att     = row.get("File Attachments", "").strip()
        auteurs = parser_auteurs(row.get("Author", ""))
        annee   = row.get("Publication Year", "").strip()
        titre   = row.get("Title", "").strip()

        pdf_src = trouver_pdf(att, zotero_storage)

        if pdf_src is None:
            sans_pdf.append({
                "key": key, "auteur": row.get("Author", ""),
                "titre": titre, "annee": annee,
            })
            continue

        nom = nom_fichier(auteurs, annee, noms_existants)
        noms_existants.add(nom)
        dest = pdf_dir / nom

        if not dry_run:
            try:
                if not dest.exists():
                    if mode == "lien":
                        dest.symlink_to(pdf_src.resolve())
                    else:
                        shutil.copy2(pdf_src, dest)
            except Exception as e:
                erreurs.append({"key": key, "titre": titre, "erreur": str(e)})
                continue

        rc = ref_courte(auteurs, annee)
        rl = ref_longue(row, auteurs)

        metadata[nom] = {
            "key"       : key,
            "auteur"    : row.get("Author", ""),
            "titre"     : titre,
            "annee"     : annee,
            "editeur"   : row.get("Publisher", ""),
            "revue"     : row.get("Publication Title", ""),
            "type_item" : row.get("Item Type", ""),
            "ref_courte": rc,
            "ref_longue": rl,
            "pdf_source": str(pdf_src),
        }
        importes.append({"key": key, "nom": nom, "ref_courte": rc})

    return metadata, importes, sans_pdf, erreurs


# =============================================================================
# RAPPORT
# =============================================================================

def generer_rapport(importes, sans_pdf, erreurs, timestamp) -> Path:
    lignes = [
        "# Rapport d'import Zotero",
        "",
        f"**Timestamp** : {timestamp}  ",
        f"**Importés** : {len(importes)}  ",
        f"**Sans PDF** : {len(sans_pdf)}  ",
        f"**Erreurs** : {len(erreurs)}  ",
        "",
        "---",
        "",
    ]
    if importes:
        lignes += ["## Importés", ""]
        for i in importes[:30]:
            lignes.append(f"- `{i['nom']}` — {i['ref_courte']}")
        if len(importes) > 30:
            lignes.append(f"- … et {len(importes)-30} autres")
        lignes.append("")
    if sans_pdf:
        lignes += ["## Sans PDF (ignorés)", ""]
        for i in sans_pdf:
            a = i['auteur'].split(';')[0].strip()[:35]
            lignes.append(f"- {a} ({i['annee']}) — {i['titre'][:55]}")
        lignes.append("")
    if erreurs:
        lignes += ["## Erreurs", ""]
        for i in erreurs:
            lignes.append(f"- ❌ {i['titre'][:55]} — {i['erreur']}")
        lignes.append("")
    lignes += [
        "---", "",
        "## Étape suivante", "",
        "```bash",
        "python 01_extract_text.py",
        "python 02_chunk_corpus.py",
        "python 03_build_embeddings.py",
        "```", "",
        "*Rapport généré par `00_zotero_import.py`.*",
    ]
    chemin = Path(f"rapport_import_zotero_{timestamp}.md")
    chemin.write_text("\n".join(lignes), encoding="utf-8")
    return chemin


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="00_zotero_import — Import initial Zotero → pdfs/ + metadata_refs.json"
    )
    parser.add_argument("--csv", type=str, default=CSV_ZOTERO)
    parser.add_argument("--dry-run", action="store_true",
                        help="Lecture seule, rien n'est créé.")
    args = parser.parse_args()

    chemin_csv     = Path(args.csv)
    zotero_storage = Path(ZOTERO_STORAGE)
    pdf_dir        = Path(PDF_DIR)
    metadata_path  = Path(METADATA_REFS)
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("  IMPORT ZOTERO — construction de pdfs/ et metadata_refs.json")
    print("=" * 60)
    print(f"  CSV     : {chemin_csv}")
    print(f"  Storage : {zotero_storage}")
    print(f"  Mode    : {'dry-run' if args.dry_run else MODE_FICHIER}")
    print()

    if not chemin_csv.exists():
        print(f"❌ CSV introuvable : {chemin_csv.resolve()}")
        sys.exit(1)

    if not args.dry_run:
        pdf_dir.mkdir(parents=True, exist_ok=True)

    metadata, importes, sans_pdf, erreurs = traiter_csv(
        chemin_csv, zotero_storage, pdf_dir, MODE_FICHIER, args.dry_run
    )

    if not args.dry_run:
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\n  ✅ metadata_refs.json : {len(metadata)} entrées")
        print(f"  ✅ pdfs/ : {len(importes)} fichiers")

    rapport = generer_rapport(importes, sans_pdf, erreurs, timestamp)

    print(f"\n{'═'*60}")
    print(f"  Importés    : {len(importes)}")
    print(f"  Sans PDF    : {len(sans_pdf)}")
    print(f"  Erreurs     : {len(erreurs)}")
    print(f"{'═'*60}")
    print(f"\n  Rapport : {rapport}\n")


if __name__ == "__main__":
    main()
