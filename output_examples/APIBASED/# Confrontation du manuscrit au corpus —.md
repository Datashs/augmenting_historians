# Confrontation du manuscrit au corpus — Exemples de signaux

Ce document présente une sélection d’exemples issus de la confrontation automatisée entre le manuscrit et le corpus.
(scripts 06,07)

⚠️ Les éléments ci-dessous ne constituent pas une interprétation historienne.
Ils exposent des **signaux produits par le pipeline** :
- alignements
- enrichissements possibles
- absences de correspondance

Les reformulations proposées sont **strictement indicatives** : elles explicitent ce que suggèrent les sorties, sans en garantir la validité.

---

## 1. Alignement + enrichissement définitionnel

**Extrait du manuscrit**  
> “L'extradition est une procédure technique. Fréquente, réglée par des centaines de traités bilatéraux…”

:contentReference[oaicite:0]{index=0}  

**Signal produit (enrichissement possible par le corpus)**  
- Définition juridique précise disponible dans le corpus  
- Inscription historique (fin XIXe – début XXe siècle) reserre la datation

**Extraits mobilisables (sortie)**  
- “L’extradition est le mécanisme juridique par lequel un État… remet un individu…”  
- “Il faudra attendre la fin du XIXe siècle pour qu’elle devienne un enjeu universel…”

**Effet observable**  
→ Le paragraphe peut être **densifié par insertion de définition et de repères temporels**

**Reformulation suggérée (indicative)**  
Le texte peut intégrer une définition juridique stabilisée et préciser le moment de généralisation internationale de l’extradition.

---

## 2. Alignement thématique + extension des acteurs

**Extrait du manuscrit**  
> “Ces histoires de l’extradition sont menées souvent en retraçant les étapes d’une affaire…”

:contentReference[oaicite:1]{index=1}  

**Signal produit (enrichissement)**  
- Ajout d’acteurs identifiés dans le corpus : juridictions, diplomates  
- Mise en évidence de débats doctrinaux

**Extraits mobilisables (sortie)**  
- “Le XIXe siècle sera celui de la confrontation des points de vue…”  
- “L’extradition politique concentre les enjeux de la répartition des compétences…”

**Effet observable**  
→ Passage d’une description centrée sur les affaires à une **pluralisation des niveaux d’analyse**

**Reformulation suggérée (indicative)**  
Le texte peut intégrer explicitement la diversité des acteurs et des niveaux de décision impliqués.

---

## 3. Alignement chronologique + sous-spécification

**Extrait du manuscrit**  
> “Au cours du XIXe siècle… la fréquence augmente…”

:contentReference[oaicite:2]{index=2}  

**Signal produit (enrichissement)**  
- Datations précises disponibles :
  - années 1830 (développement des traités)
  - 1869 (extension aux délits)

**Extraits mobilisables (sortie)**  
- “À partir de 1830, le droit extraditionnel se perfectionne…”  
- “Il faut arriver à 1869… pour que les délits apparaissent…”

**Effet observable**  
→ Le passage peut être **spécifié par des jalons datés**

**Reformulation suggérée (indicative)**  
Le texte peut remplacer une tendance générale par une séquence datée.

---

## 4. Signal réflexif confirmé + explicitation des biais

**Extrait du manuscrit**  
> “Ces instruments ont leurs limites… les biais qu’elles introduisent…”

:contentReference[oaicite:3]{index=3}  

**Signal produit (enrichissement)**  
- Explicitation des biais :
  - dépendance aux choix de catégorisation
  - décontextualisation des unités textuelles

**Extraits mobilisables (sortie)**  
- “Les instruments… ne sont pas objectifs…”  
- “Le traitement suppose de réduire les textes en unités comparables…”

**Effet observable**  
→ Le passage peut être **renforcé par une explicitation technique des biais**

**Reformulation suggérée (indicative)**  
Le texte peut préciser les mécanismes concrets de production des biais.

---

## 5. Intuition non documentée (signal de manque)

**Extrait du manuscrit**  
> “L’histoire de l’extradition… est rythmée par quelques grandes affaires…”

:contentReference[oaicite:4]{index=4}  

**Signal produit (manque)**  
- Aucune affaire précise proposée dans les sorties  
- Suggestion explicite d’ajouter des exemples documentés dans le corpus

**Effet observable**  
→ Le passage reste **non étayé empiriquement**

**Sortie associée**  
- “Des exemples précis d’affaires… seraient nécessaires”

**Reformulation suggérée (indicative)**  
Le texte peut être complété par des cas empiriques identifiables.

---

## 6. Absence d’alignement détectée

**Extrait du manuscrit**  
> “Il faut s’arrêter sur la façon dont cette procédure a existé dans l’espace public…”

:contentReference[oaicite:5]{index=5}  

**Signal produit (absence)**  
- Aucun passage directement mobilisable identifié  
- Aucun extrait aligné proposé

**Effet observable**  
→ Zone **non couverte par le corpus actuel**

**Interprétation minimale autorisée**  
→ Le passage n’est ni confirmé ni enrichi par les sources disponibles. 
Logique en ce cas, les traités juridiques anciens ne mentionnent pas la 
fortune médiatique des affaires d'extradition. 

---

## Synthèse des signaux observés

- Les sorties produisent majoritairement :
  - des **compléments (définitions, dates, citations)**
  - des **extensions (acteurs, niveaux d’analyse)**
  - des **alertes de manque (absence d’exemples)**

- Les effets principaux sont :
  - densification locale des paragraphes
  - explicitation de tendances implicites
  - identification de zones non couvertes

- Les sorties ne produisent pas :
  - de structuration globale du raisonnement
  - de hiérarchisation des arguments
  - de validation interprétative

---

## Remarque méthodologique

Ce document expose des **effets de lecture assistée** produits par le pipeline.

Toute interprétation historienne doit :
- être explicitement distinguée de ces signaux  
- être produite en dehors de cette couche
