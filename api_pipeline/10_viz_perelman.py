#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10_viz_perelman.py
==================
Génère un rapport HTML interactif à partir du JSON produit par
10_map_perelman_openai.py.

Le fichier HTML est auto-contenu (JS/CSS embarqués, aucune dépendance réseau)
et s'ouvre directement dans le navigateur.

Position dans le pipeline :
    10_map_perelman_openai.py → perelman_{ts}.json
    10_viz_perelman.py        → perelman_{ts}_viz.html  ← ce script

UTILISATION :
    python 10_viz_perelman.py
    python 10_viz_perelman.py resultats/perelman_20260416_120000.json
    python 10_viz_perelman.py --no-open
"""

# =============================================================================
# PARAMÈTRES
# =============================================================================

JSON_DIR   = "resultats"
OUTPUT_DIR = "resultats"
AUTO_OPEN  = True

# =============================================================================
# IMPORTS
# =============================================================================

import argparse
import json
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

# =============================================================================
# DONNÉES DE RÉFÉRENCE
# =============================================================================

FAMILLES = {
    "quasi_logique" : {"label": "A1 — quasi-logiques",        "color": "#185FA5"},
    "structure_reel": {"label": "A2 — structure du réel",      "color": "#0F6E56"},
    "fonde_reel"    : {"label": "A3 — fondent la structure",   "color": "#3B6D11"},
    "dissociation"  : {"label": "B  — dissociation",           "color": "#854F0B"},
}
FAMILLES_ORDER = ["quasi_logique", "structure_reel", "fonde_reel", "dissociation"]

AUD_DESC = {
    "universel"                : "prétend s'adresser à tout être raisonnable",
    "particulier_disciplinaire": "communauté des historiens (conventions épistémiques)",
    "particulier_ideologique"  : "groupe aux valeurs communes",
    "indetermine"              : "auditoire non clairement construit",
}

# =============================================================================
# UTILITAIRES
# =============================================================================

def trouver_json(json_dir: Path) -> Path:
    fichiers = sorted(json_dir.glob("perelman_*.json"), reverse=True)
    if not fichiers:
        raise FileNotFoundError(
            f"Aucun fichier perelman_*.json dans {json_dir}.\n"
            "Lancez d'abord 10_map_perelman_openai.py."
        )
    return fichiers[0]


def charger_json(chemin: Path) -> dict:
    with open(chemin, encoding="utf-8") as f:
        return json.load(f)


def profil_color(v: float) -> str:
    if v >= 0.7:
        return "#3B6D11"
    if v >= 0.5:
        return "#BA7517"
    return "#A32D2D"


def extraire_synthese(analyse_brute: str) -> str:
    if not analyse_brute:
        return "(non disponible)"
    m = re.search(
        r"(?:SYNTHÈSE RHÉTORIQUE|synthèse rhétorique|SYNTHÈSE|synthèse)[^\n]*\n(.+?)(?:\n\s*━|$)",
        analyse_brute,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return analyse_brute.strip()[-600:]


def esc(s: str) -> str:
    return (str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# =============================================================================
# HEATMAP — données tabulaires pour injection HTML
# =============================================================================

def build_heatmap_html(paras: list) -> str:
    if not paras:
        return ""

    # Maximum par famille pour normalisation de la couleur
    max_fam = {}
    for f in FAMILLES_ORDER:
        max_fam[f] = max(1, max(len((p.get("techniques_detectees") or {}).get(f, [])) for p in paras))

    def hm_color(ratio: float) -> str:
        stops = ["#E6F1FB", "#B5D4F4", "#85B7EB", "#378ADD", "#185FA5", "#042C53"]
        if ratio == 0:
            return "#f1efe8"
        idx = min(5, int(ratio * 6))
        return stops[idx]

    def text_color(ratio: float) -> str:
        return "#ffffff" if ratio > 0.6 else "#185FA5"

    # En-tête
    headers = "".join(
        f'<th style="text-align:center;color:{FAMILLES[f]["color"]};white-space:nowrap">'
        f'{FAMILLES[f]["label"].split("—")[0].strip()}</th>'
        for f in FAMILLES_ORDER
    )
    html = (f'<table class="hm-table">'
            f'<thead><tr><th class="hm-sec-th">section</th>'
            f'{headers}'
            f'<th style="text-align:center">profil</th></tr></thead><tbody>')

    for p in paras:
        titre = esc((p.get("titre") or "")[:32])
        titre_esc = esc(p.get("titre", ""))
        pid = p["id"]
        html += f'<tr><td class="hm-sec-td" title="{titre_esc}">§{pid} {titre}</td>'
        for f in FAMILLES_ORDER:
            techs = (p.get("techniques_detectees") or {}).get(f, [])
            n = len(techs)
            ratio = n / max_fam[f]
            bg = hm_color(ratio)
            tc = text_color(ratio)
            tip = ", ".join(techs) if techs else "—"
            cell = (f'<span style="font-size:11px;font-family:monospace;'
                    f'font-weight:500;color:{tc}">{n}</span>') if n else ""
            html += (f'<td style="padding:2px 4px" title="{esc(tip)}">'
                     f'<div class="hm-cell" style="background:{bg};'
                     f'display:flex;align-items:center;justify-content:center">'
                     f'{cell}</div></td>')
        pv = p.get("score_profil_argumentatif", 0)
        html += (f'<td style="text-align:center;font-family:monospace;font-size:12px;'
                 f'font-weight:500;color:{profil_color(pv)}">{pv:.2f}</td></tr>')

    html += "</tbody></table>"
    return html


# =============================================================================
# DONNÉES JS (version allégée — sans le texte brut LLM)
# =============================================================================

def build_data_js(paras: list) -> str:
    slim = []
    for p in paras:
        slim.append({
            "id"                                   : p.get("id"),
            "titre"                                : p.get("titre", ""),
            "score_force_persuasive"               : p.get("score_force_persuasive", 0),
            "score_ancrage_auditoire"              : p.get("score_ancrage_auditoire", 0),
            "score_coherence_valeurs"              : p.get("score_coherence_valeurs", 0),
            "score_risque_sophistique_rhethorique" : p.get("score_risque_sophistique_rhethorique", 0),
            "score_profil_argumentatif"            : p.get("score_profil_argumentatif", 0),
            "technique_dominante"                  : p.get("technique_dominante", ""),
            "type_auditoire"                       : p.get("type_auditoire", "indetermine"),
            "techniques_detectees"                 : p.get("techniques_detectees", {}),
            "paires_dissociation"                  : p.get("paires_dissociation", []),
            "valeurs_mobilisees"                   : p.get("valeurs_mobilisees", []),
            "usages_sophistiques"                  : p.get("usages_sophistiques", []),
            "synthese"                             : extraire_synthese(p.get("analyse", "")),
        })
    return json.dumps(slim, ensure_ascii=False)


# =============================================================================
# SECTION CARDS
# =============================================================================

def build_section_cards(paras: list) -> str:
    cards = []
    for p in paras:
        pv   = p.get("score_profil_argumentatif", 0)
        col  = profil_color(pv)
        pct  = round(pv * 100)
        tech = (p.get("technique_dominante") or "—").replace("_", " ")
        aud  = (p.get("type_auditoire") or "indetermine").replace("_", " ")
        sophs = p.get("usages_sophistiques") or []
        soph_html = (f'<div class="sc-soph">⚑ {esc(", ".join(sophs))}</div>'
                     if sophs else "")
        cards.append(f"""
        <div class="sec-card" id="sc{p['id']}" onclick="selectSection({p['id']})">
          <div class="sec-id">§ {p['id']}</div>
          <div class="sec-title">{esc((p.get('titre') or '')[:50])}</div>
          <div class="prof-bar-wrap">
            <div class="prof-bar-bg">
              <div class="prof-bar-fill" style="width:{pct}%;"></div>
            </div>
            <div class="prof-val" style="color:{col}">{pv:.2f}</div>
          </div>
          <span class="tech-tag">{esc(tech)}</span>
          <div class="sc-aud">{esc(aud)}</div>
          {soph_html}
        </div>""")
    return "\n".join(cards)


# =============================================================================
# GÉNÉRATION HTML
# =============================================================================

def generer_html(data: dict, chemin_json: Path) -> str:
    paras      = data.get("paragraphes", data.get("sections", []))
    score_glob = data.get("score_global", 0)
    profil     = data.get("profil_dominant", "")
    modele     = data.get("modele_llm", "")
    ts         = data.get("timestamp", "")
    nb         = len(paras)

    force_moy   = sum(p.get("score_force_persuasive", 0) for p in paras) / nb if nb else 0
    ancrage_moy = sum(p.get("score_ancrage_auditoire", 0) for p in paras) / nb if nb else 0
    soph_n      = sum(1 for p in paras if (p.get("usages_sophistiques") or []))
    aud_map: dict = {}
    for p in paras:
        a = p.get("type_auditoire", "indetermine")
        aud_map[a] = aud_map.get(a, 0) + 1
    aud_dom = max(aud_map, key=aud_map.get) if aud_map else "—"

    heatmap_html   = build_heatmap_html(paras)
    section_cards  = build_section_cards(paras)
    data_js        = build_data_js(paras)
    familles_js    = json.dumps(FAMILLES)
    familles_order_js = json.dumps(FAMILLES_ORDER)
    aud_desc_js    = json.dumps(AUD_DESC)
    prof_col_glob  = profil_color(score_glob)

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Analyse rhétorique Perelman — {esc(ts)}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:system-ui,sans-serif;background:#f5f4f0;color:#1a1a18;font-size:15px;line-height:1.6}}
.page{{max-width:1100px;margin:0 auto;padding:1.5rem}}
h1{{font-size:20px;font-weight:500;margin-bottom:0.25rem}}
.subtitle{{font-size:13px;color:#5f5e5a;font-family:monospace;margin-bottom:1.5rem}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:1.5rem}}
.mc{{background:#fff;border-radius:8px;padding:.75rem 1rem;border:0.5px solid #d3d1c7}}
.mc .lbl{{font-size:12px;color:#888780;margin-bottom:3px}}
.mc .val{{font-size:22px;font-weight:500}}
.mc .sub{{font-size:11px;color:#b4b2a9;margin-top:2px;font-family:monospace}}
.hm-wrap{{background:#fff;border-radius:8px;border:0.5px solid #d3d1c7;padding:1rem;margin-bottom:1.5rem;overflow-x:auto}}
.hm-title{{font-size:13px;color:#888780;margin-bottom:.75rem;font-weight:500}}
.hm-table{{width:100%;border-collapse:collapse;font-size:12px}}
.hm-table th{{font-size:11px;font-weight:500;color:#888780;padding:4px 6px;text-align:left;border-bottom:0.5px solid #d3d1c7;font-family:monospace}}
.hm-sec-th{{width:140px}}
.hm-table td{{padding:2px 4px;border:0.5px solid #e8e6df}}
.hm-sec-td{{font-family:monospace;font-size:11px;color:#888780;white-space:nowrap;max-width:140px;overflow:hidden;text-overflow:ellipsis}}
.hm-cell{{width:100%;height:24px;border-radius:3px}}
.hm-legend{{display:flex;gap:4px;align-items:center;margin-top:8px;font-size:11px;color:#888780}}
.grid-label{{font-size:13px;color:#888780;margin-bottom:0.5rem}}
.sec-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:10px;margin-bottom:1.5rem}}
.sec-card{{background:#fff;border:0.5px solid #d3d1c7;border-radius:8px;padding:.75rem;cursor:pointer;transition:border-color .15s}}
.sec-card:hover{{border-color:#185FA5}}
.sec-card.active{{border:1.5px solid #185FA5}}
.sec-id{{font-size:11px;font-family:monospace;color:#b4b2a9}}
.sec-title{{font-size:13px;font-weight:500;margin:2px 0 5px;line-height:1.3}}
.prof-bar-wrap{{display:flex;align-items:center;gap:6px;margin-bottom:4px}}
.prof-bar-bg{{flex:1;height:4px;background:#d3d1c7;border-radius:2px}}
.prof-bar-fill{{height:4px;border-radius:2px;background:#185FA5}}
.prof-val{{font-size:12px;font-family:monospace;width:32px;text-align:right}}
.tech-tag{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;border:0.5px solid #d3d1c7;color:#888780}}
.sc-aud{{font-size:11px;color:#b4b2a9;margin-top:3px}}
.sc-soph{{font-size:11px;color:#BA7517;margin-top:3px}}
.detail{{display:none;background:#fff;border:0.5px solid #d3d1c7;border-radius:10px;padding:1.25rem;margin-bottom:1.5rem}}
.detail.visible{{display:block}}
.d-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:1rem}}
.d-title{{font-size:16px;font-weight:500}}
.d-id{{font-size:12px;font-family:monospace;color:#b4b2a9}}
.close-btn{{background:none;border:none;cursor:pointer;color:#888780;font-size:18px;padding:0}}
.scores-row{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:1.25rem}}
.sp{{display:flex;flex-direction:column;align-items:center;border:0.5px solid #d3d1c7;border-radius:8px;padding:.5rem .75rem;min-width:80px}}
.sp .spl{{font-size:11px;color:#888780;text-align:center;margin-bottom:4px}}
.sp .spv{{font-size:18px;font-weight:500}}
.tabs{{display:flex;border-bottom:0.5px solid #d3d1c7;margin-bottom:1rem;flex-wrap:wrap}}
.tab-btn{{background:none;border:none;cursor:pointer;padding:6px 14px;font-size:13px;color:#888780;border-bottom:2px solid transparent}}
.tab-btn.active{{color:#1a1a18;border-bottom-color:#185FA5}}
.tc{{display:none}}.tc.active{{display:block}}
.fam-block{{margin-bottom:1rem}}
.fam-title{{font-size:12px;font-weight:500;font-family:monospace;margin-bottom:6px;padding-bottom:4px;border-bottom:0.5px solid #e8e6df}}
.tech-pills{{display:flex;flex-wrap:wrap;gap:6px}}
.tech-pill{{padding:3px 10px;border-radius:10px;font-size:12px;font-family:monospace;border:0.5px solid #d3d1c7}}
.tp-a1{{border-color:#378ADD;color:#185FA5}}
.tp-a2{{border-color:#1D9E75;color:#0F6E56}}
.tp-a3{{border-color:#639922;color:#3B6D11}}
.tp-b {{border-color:#EF9F27;color:#854F0B}}
.tech-none{{font-size:13px;color:#b4b2a9;font-style:italic}}
.diss-list{{list-style:none;display:flex;flex-direction:column;gap:6px}}
.diss-list li{{font-size:13px;padding:6px 10px;border-left:2px solid #EF9F27;background:#faeeda;border-radius:0 4px 4px 0}}
.val-pills{{display:flex;flex-wrap:wrap;gap:6px}}
.val-pill{{padding:3px 10px;background:#f1efe8;border-radius:10px;font-size:12px}}
.aud-type{{font-size:14px;font-weight:500;margin-bottom:4px}}
.aud-desc{{font-size:13px;color:#888780}}
.soph-item{{font-size:13px;color:#A32D2D;padding:3px 0}}
.soph-ok{{font-size:13px;color:#3B6D11}}
.synth-text{{font-size:14px;line-height:1.7;white-space:pre-wrap}}
footer{{font-size:12px;color:#b4b2a9;text-align:center;margin-top:2rem;font-family:monospace}}
@media(max-width:600px){{.hm-wrap{{font-size:11px}}}}
</style>
</head>
<body>
<div class="page">
  <h1>Analyse rhétorique — Nouvelle Rhétorique (Perelman)</h1>
  <div class="subtitle">{esc(modele)} · {esc(ts)} · source : {esc(chemin_json.name)}</div>

  <div class="metrics">
    <div class="mc">
      <div class="lbl">profil global</div>
      <div class="val" style="color:{prof_col_glob}">{score_glob:.2f}</div>
      <div class="sub">score moyen</div>
    </div>
    <div class="mc">
      <div class="lbl">technique dominante</div>
      <div class="val" style="font-size:12px;padding-top:6px;font-family:monospace;line-height:1.3">
        {esc((profil or "—").replace("_"," "))}
      </div>
    </div>
    <div class="mc">
      <div class="lbl">force persuasive moy.</div>
      <div class="val">{force_moy:.2f}</div>
      <div class="sub">/1.00</div>
    </div>
    <div class="mc">
      <div class="lbl">ancrage auditoire moy.</div>
      <div class="val">{ancrage_moy:.2f}</div>
      <div class="sub">/1.00</div>
    </div>
    <div class="mc">
      <div class="lbl">auditoire dominant</div>
      <div class="val" style="font-size:12px;padding-top:6px;line-height:1.3">
        {esc(aud_dom.replace("_"," "))}
      </div>
    </div>
    <div class="mc">
      <div class="lbl">usages sophistiques</div>
      <div class="val" style="color:{'#BA7517' if soph_n > 0 else '#3B6D11'}">{soph_n}</div>
      <div class="sub">sections concernées</div>
    </div>
  </div>

  <div class="hm-wrap">
    <div class="hm-title">présence des techniques par section — familles A1 / A2 / A3 / B</div>
    {heatmap_html}
    <div class="hm-legend">
      densité :
      {''.join(f'<div style="width:16px;height:12px;background:{c};border-radius:2px;"></div>' for c in ["#f1efe8","#E6F1FB","#85B7EB","#378ADD","#185FA5","#042C53"])}
      0 → max
    </div>
  </div>

  <div class="grid-label">sections — cliquer pour le détail</div>
  <div class="sec-grid">{section_cards}</div>
  <div class="detail" id="detailPane"></div>
</div>

<footer>généré par 10_viz_perelman.py · Perelman &amp; Olbrechts-Tyteca (1958)</footer>

<script>
const PARAS    = {data_js};
const FAMILLES = {familles_js};
const FAM_ORDER= {familles_order_js};
const AUD_DESC = {aud_desc_js};
const FAM_CLS  = {{quasi_logique:'tp-a1',structure_reel:'tp-a2',fonde_reel:'tp-a3',dissociation:'tp-b'}};
let activeId = null;
let activeTab = 'techniques';

function profColor(v){{
  if(v>=0.7) return '#3B6D11';
  if(v>=0.5) return '#BA7517';
  return '#A32D2D';
}}

function selectSection(id){{
  const p=PARAS.find(x=>x.id===id);
  if(!p) return;
  if(activeId===id){{
    document.getElementById('detailPane').classList.remove('visible');
    document.querySelectorAll('.sec-card').forEach(c=>c.classList.remove('active'));
    activeId=null; return;
  }}
  activeId=id;
  document.querySelectorAll('.sec-card').forEach(c=>c.classList.remove('active'));
  const card=document.getElementById('sc'+id);
  if(card){{card.classList.add('active'); card.scrollIntoView({{behavior:'smooth',block:'nearest'}});}}
  renderDetail(p);
  document.getElementById('detailPane').classList.add('visible');
}}

function renderDetail(p){{
  const pv=p.score_profil_argumentatif||0;
  const scoreItems=[
    ['Force',p.score_force_persuasive,'#185FA5'],
    ['Ancrage',p.score_ancrage_auditoire,'#0F6E56'],
    ['Valeurs',p.score_coherence_valeurs,'#3B6D11'],
    ['Risque',p.score_risque_sophistique_rhethorique,'#BA7517'],
    ['Profil',pv,profColor(pv)],
  ];

  const techHtml=FAM_ORDER.map(fam=>{{
    const items=(p.techniques_detectees||{{}})[fam]||[];
    const info=FAMILLES[fam];
    const cls=FAM_CLS[fam];
    return `<div class="fam-block">
      <div class="fam-title" style="color:${{info.color}}">${{info.label}}</div>
      <div class="tech-pills">${{items.length
        ? items.map(t=>`<span class="tech-pill ${{cls}}">${{t.replace(/_/g,' ')}}</span>`).join('')
        : '<span class="tech-none">aucune technique</span>'
      }}</div></div>`;
  }}).join('');

  const pairesHtml=(p.paires_dissociation||[]).length
    ?`<ul class="diss-list">${{(p.paires_dissociation).map(x=>`<li>${{escHtml(x)}}</li>`).join('')}}</ul>`
    :`<div class="tech-none">aucune paire identifiée</div>`;

  const valeursHtml=(p.valeurs_mobilisees||[]).length
    ?`<div class="val-pills">${{p.valeurs_mobilisees.map(v=>`<span class="val-pill">${{escHtml(v)}}</span>`).join('')}}</div>`
    :`<div class="tech-none">aucune valeur identifiée</div>`;

  const aud=p.type_auditoire||'indetermine';
  const audHtml=`<div class="aud-type">${{aud.replace(/_/g,' ')}}</div>
    <div class="aud-desc">${{AUD_DESC[aud]||''}}</div>`;

  const sophHtml=(p.usages_sophistiques||[]).length
    ?(p.usages_sophistiques.map(s=>`<div class="soph-item">⚑ ${{s.replace(/_/g,' ')}}</div>`).join(''))
    :`<div class="soph-ok">aucun usage sophistique détecté</div>`;

  document.getElementById('detailPane').innerHTML=`
    <div class="d-header">
      <div><div class="d-id">§ ${{p.id}}</div>
        <div class="d-title">${{escHtml(p.titre)}}</div></div>
      <button class="close-btn" onclick="selectSection(${{p.id}})">✕</button>
    </div>
    <div class="scores-row">
      ${{scoreItems.map(([l,v,c])=>`<div class="sp"><div class="spl">${{l}}</div>
        <div class="spv" style="color:${{c}}">${{(v||0).toFixed(2)}}</div></div>`).join('')}}
    </div>
    <div class="tabs">
      <button class="tab-btn${{activeTab==='techniques'?' active':''}}" onclick="switchTab('techniques',this)">Techniques</button>
      <button class="tab-btn${{activeTab==='dissociation'?' active':''}}" onclick="switchTab('dissociation',this)">Dissociation</button>
      <button class="tab-btn${{activeTab==='auditoire'?' active':''}}" onclick="switchTab('auditoire',this)">Auditoire</button>
      <button class="tab-btn${{activeTab==='valeurs'?' active':''}}" onclick="switchTab('valeurs',this)">Valeurs</button>
      <button class="tab-btn${{activeTab==='sophismes'?' active':''}}" onclick="switchTab('sophismes',this)">Sophismes</button>
      <button class="tab-btn${{activeTab==='synthese'?' active':''}}" onclick="switchTab('synthese',this)">Synthèse</button>
    </div>
    <div id="tc-techniques"  class="tc${{activeTab==='techniques'?' active':''}}">${{techHtml}}</div>
    <div id="tc-dissociation" class="tc${{activeTab==='dissociation'?' active':''}}">${{pairesHtml}}</div>
    <div id="tc-auditoire"   class="tc${{activeTab==='auditoire'?' active':''}}">${{audHtml}}</div>
    <div id="tc-valeurs"     class="tc${{activeTab==='valeurs'?' active':''}}">${{valeursHtml}}</div>
    <div id="tc-sophismes"   class="tc${{activeTab==='sophismes'?' active':''}}">${{sophHtml}}</div>
    <div id="tc-synthese"    class="tc${{activeTab==='synthese'?' active':''}}">
      <div class="synth-text">${{escHtml(p.synthese||'(non disponible)')}}</div></div>
  `;
}}

function switchTab(tab,btn){{
  activeTab=tab;
  document.querySelectorAll('#detailPane .tab-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#detailPane .tc').forEach(c=>c.classList.remove('active'));
  const el=document.getElementById('tc-'+tab); if(el) el.classList.add('active');
}}

function escHtml(s){{
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}
</script>
</body>
</html>"""


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="10_viz_perelman — Génère un rapport HTML depuis un JSON du script 10."
    )
    parser.add_argument("fichier", nargs="?", default=None,
                        help="Fichier JSON perelman_*.json. "
                             "Si absent : prend le plus récent dans JSON_DIR.")
    parser.add_argument("--no-open", action="store_true",
                        help="Génère le HTML sans ouvrir le navigateur.")
    args = parser.parse_args()

    json_dir   = Path(JSON_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fichier:
        chemin_json = Path(args.fichier)
        if not chemin_json.is_absolute():
            chemin_json = json_dir / chemin_json
    else:
        print("Aucun fichier spécifié — recherche du plus récent…")
        chemin_json = trouver_json(json_dir)

    if not chemin_json.exists():
        print(f"❌ Fichier introuvable : {chemin_json}")
        sys.exit(1)

    print(f"Chargement : {chemin_json.name}")
    data  = charger_json(chemin_json)
    paras = data.get("paragraphes", data.get("sections", []))
    print(f"  {len(paras)} paragraphe(s) · score global : {data.get('score_global', 0):.3f}")

    ts        = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    html_path = output_dir / f"perelman_{ts}_viz.html"
    html      = generer_html(data, chemin_json)
    html_path.write_text(html, encoding="utf-8")

    print(f"✅ HTML généré : {html_path}")
    print(f"   Taille : {len(html):,} octets")

    if AUTO_OPEN and not args.no_open:
        webbrowser.open(html_path.resolve().as_uri())
        print("   Navigateur ouvert.")


if __name__ == "__main__":
    main()
