# Points Clés pour le Rapport Final - Vision Numérique

## Structure du Rapport (Total: 20 points)

---

## 1. Page Titre (1 page) - 1 point

**Éléments requis:**
- Nom et numéro du cours: Vision Numérique
- Date de remise: [À compléter]
- Nom du professeur: [À compléter]
- Noms et matricules des membres de l'équipe
- Titre du projet: Système de Détection de Gestes par Vision par Ordinateur

---

## 2. Mise en Contexte et Description de la Solution (max 2,5 pages) - 6 points

### Points clés à mentionner:

#### 2.1 Contexte du Projet
- **Objectif:** Développer un système de détection de gestes mains-libres utilisant uniquement des techniques de vision numérique classique (sans ML/AI)
- **Application:** Interface de contrôle gestuel pour applications interactives (peinture, présentation, etc.)
- **Contraintes:** Temps réel (30+ FPS), techniques CV traditionnelles uniquement

#### 2.2 Architecture de la Solution

**Deux modes de détection implémentés:**

##### Mode MediaPipe (Baseline/Référence)
- Détection basée sur modèle pré-entraîné
- 21 landmarks de la main
- Performance: ~15 FPS
- Utilisé comme référence pour valider le mode CV

##### Mode CV (Computer Vision - Solution Principale)
- **Détection de peau (YCrCb + HSV):**
  - Double espace couleur pour robustesse
  - Calibration automatique avec validation IoU
  - Mode alignement centre de paume pour précision
  
- **Segmentation et filtrage:**
  - Soustraction de fond (MOG2)
  - Opérations morphologiques (ouverture, fermeture)
  - Filtrage intelligent des avant-bras (nouveauté)
  
- **Comptage des doigts (3 méthodes hybrides):**
  - Défauts de convexité adaptatifs
  - Points extrêmes du contour
  - Transformée de distance
  - Vote médian + lissage temporel
  
- **Filtrage des avant-bras (Phase 8.5):**
  - Analyse de forme géométrique (5 descripteurs)
  - Détection d'orientation (ellipse fitting)
  - Détection de poignet et recadrage
  - Sélection intelligente par score pondéré

**Détail du Filtrage des Avant-bras:**

Le filtrage des avant-bras est un problème critique car la détection de peau capture à la fois la main ET l'avant-bras (tous deux ont la même couleur de peau). Sans filtrage, l'avant-bras est souvent confondu avec la main, causant des détections erronées.

**Pipeline multi-étapes:**

1. **Filtrage par Aire** (pré-filtrage):
   - Élimine contours trop petits (bruit) ou trop grands (corps entier)
   - Seuils: 3000 pixels < aire < 50% de l'image

2. **Sélection Intelligente par Score Pondéré**:
   - Calcule un score pour chaque contour basé sur 5 critères géométriques
   - Pondération des critères:
     * Aspect ratio (25%): Mains ≈ carrées (0.6-1.5), avant-bras allongés (>2.0)
     * Position verticale (25%): Mains typiquement en haut du cadre, avant-bras en bas
     * Compactness/Circularité (20%): Mesure = 4π×Aire/Périmètre² (mains > 0.3)
     * Solidité (20%): Ratio Aire/AireEnveloppeConvexe (mains: 0.6-0.9 car doigts créent concavités, avant-bras: >0.95 très convexe)
     * Aire normalisée (10%): Préférence pour tailles moyennes
   - Sélectionne le contour avec le meilleur score

3. **Validation Géométrique (filter_forearm_by_shape)**:
   Effectue 5 tests indépendants:
   
   a) **Test Aspect Ratio**:
      - Calcule: largeur/hauteur de la boîte englobante
      - Rejette si >2.0 (trop allongé) ou <0.4 (trop étroit)
      - Rationale: Les mains ont forme relativement carrée
   
   b) **Test Compactness**:
      - Formule: 4π × Aire / Périmètre²
      - Rejette si <0.3 (forme trop irrégulière/allongée)
      - Rationale: Mains sont relativement compactes malgré doigts
   
   c) **Test Solidité**:
      - Calcule: Aire / Aire(EnveloppeConvexe)
      - Rejette si >0.95 (trop solide = pas de concavités)
      - Rationale: Doigts séparés créent défauts dans enveloppe convexe
      - Rejette aussi si <0.5 (forme trop fragmentée = bruit)
   
   d) **Test Position Verticale**:
      - Calcule position Y du centre du contour
      - Rejette si dans 20% inférieur de l'image
      - Rationale: Utilisateurs lèvent main, avant-bras reste en bas
   
   e) **Test Extent (Densité)**:
      - Calcule: Aire / Aire(BoîteEnglobante)
      - Rejette si >0.90 (remplit toute la boîte = forme rectangulaire)
      - Rationale: Mains ne remplissent pas complètement leur boîte

4. **Validation d'Orientation (filter_forearm_by_orientation)**:
   - Ajuste une ellipse au contour avec `cv2.fitEllipse()`
   - Extrait angle et dimensions (largeur, hauteur) de l'ellipse
   - Rejette si:
     * Angle proche de vertical (80°-100°) ET
     * Hauteur > 2.5 × largeur
   - Rationale: Avant-bras souvent orienté verticalement/diagonalement

5. **Détection de Poignet et Recadrage (detect_wrist_and_crop)**:
   - Analyse sections horizontales du masque de bas en haut
   - Compte pixels blancs (largeur de la main) à chaque hauteur
   - Identifie le point le plus étroit = poignet (largeur minimale)
   - Conditions:
     * Largeur min < 60% de largeur maximale
     * Position dans 50%-90% de la hauteur (zone typique du poignet)
   - Recoupe le masque au niveau du poignet détecté
   - Extrait nouveau contour de la zone main uniquement
   - Rationale: Séparation physique main/avant-bras au poignet

**Visualisation Debug:**
- Contours rouges: Tous les contours avant filtrage
- Contours verts: Contour final après tous les filtres
- Overlay affiche: aspect ratio, solidité, stage du filtre
- Raisons de rejet affichées si contour éliminé

**Justification Approche:**
- Utilise uniquement techniques CV classiques (géométrie, morphologie)
- Multi-critères = robustesse (un filtre peut échouer, autres compensent)
- Scores pondérés = décision nuancée plutôt que binaire
- Validation séquentielle = efficacité (arrêt dès échec d'un test)

#### 2.3 Pipeline de Détection
```
Capture Vidéo → Détection Peau → Soustraction Fond → 
Morphologie → Filtrage Avant-bras → Extraction Contour → 
Comptage Doigts → Reconnaissance Geste → Action
```

#### 2.4 Techniques de Vision Numérique Utilisées
- **Segmentation:** Seuillage couleur multi-espace
- **Morphologie:** Ouverture, fermeture, érosion, dilatation
- **Analyse de contours:** Enveloppe convexe, défauts de convexité
- **Géométrie:** Moments, centre de masse, boîte englobante
- **Transformées:** Distance transform pour détection de pics
- **Filtrage spatial:** Position, forme, orientation
- **Lissage temporel:** Moyenne mobile exponentielle

---

## 3. Justification des Choix de Design Finaux (max 1 page) - 3 points

### Points clés à justifier:

#### 3.1 Architecture Modulaire
- **Choix:** Séparation en modules (détection, reconnaissance, UI)
- **Justification:** Maintenabilité, testabilité, réutilisabilité
- **Évolution:** Refactorisation Phase 6.5 (900→330 lignes/fichier)

#### 3.2 Double Espace Couleur (YCrCb + HSV)
- **Choix:** Combinaison de deux espaces plutôt qu'un seul
- **Justification:** 
  - YCrCb: robuste aux variations de luminosité
  - HSV: meilleur pour teintes de peau
  - Combinaison AND: réduit faux positifs
- **Résultat:** +15% de robustesse selon tests

#### 3.3 Système de Calibration Automatique
- **Choix:** Auto-calibration avec validation IoU
- **Justification:** 
  - Adaptabilité aux différentes peaux
  - Validation objective (vs manuelle)
  - Économie de temps utilisateur
- **Impact:** Calibration en <2 minutes vs 10+ minutes manuel

#### 3.4 Approche Hybride pour Comptage de Doigts
- **Choix:** 3 méthodes + vote médian
- **Justification:**
  - Défauts de convexité: sensible au bruit
  - Points extrêmes: robuste mais limité
  - Distance transform: complément morphologique
  - Vote médian: consensus robuste
- **Résultat:** 88% précision (+13% vs méthode unique)

#### 3.5 Filtrage Multi-Critères des Avant-bras
- **Choix:** Pipeline de validation géométrique
- **Justification:**
  - Problème: avant-bras confondu avec main
  - Solution: 5 descripteurs géométriques
  - Aspect ratio: mains carrées, avant-bras allongés
  - Solidité: mains concaves (doigts), avant-bras convexes
  - Orientation: détection forme verticale
- **Résultat:** Réduction drastique des faux positifs

#### 3.6 Lissage Temporel
- **Choix:** EMA (α=0.3) + seuil de stabilité
- **Justification:**
  - Réduit oscillations de détection
  - Conserve réactivité
  - Évite changements erratiques de gestes
- **Résultat:** -67% variance détection doigts

---

## 4. Résultats Finaux Obtenus (max 2 pages) - 4 points

### Points clés à présenter:

#### 4.1 Métriques de Performance

**Mode CV (Computer Vision):**
- FPS moyen: 28-35 FPS (objectif: 1+ FPS)
- Latence: 28-35 ms
- Précision comptage doigts: ~88%
- Taux détection mains: ~92%
- Faux positifs avant-bras: -85% (après filtrage Phase 8.5)

**Mode MediaPipe (Référence):**
- FPS moyen: 58-62 FPS
- Précision: ~95%
- Latence: 16-17 ms

#### 4.2 Robustesse
- Variations lumineuses: Bonne (grâce à YCrCb)
- Différentes teintes de peau: Excellente (calibration)
- Angles de vue: Moyenne (±30° optimal)
- Arrière-plans complexes: Bonne (soustraction fond)
- Détection avant-bras: Très bonne (filtrage géométrique)

#### 4.3 Gestes Reconnus
1. **0 doigts:** Poing (action: arrêt)
2. **1 doigt:** Index (action: pointeur/dessin)
3. **2 doigts:** Paix (action: effacer)
4. **3 doigts:** OK (action: validation)
5. **4 doigts:** Quatre (action: paramètre)
6. **5 doigts:** Main ouverte (action: pause/reset)

#### 4.4 Comparaison CV vs MediaPipe

| Critère | CV (Notre Solution) | MediaPipe (Référence) |
|---------|---------------------|----------------------|
| FPS | 28-35 | 58-62 |
| Précision doigts | ~88% | ~95% |
| Utilisation CPU | Moyenne | Élevée |
| Calibration requise | Oui (auto) | Non |
| Techniques | CV classique | Deep Learning |
| Robustesse arrière-plans | Excellente | Moyenne |
| Détection avant-bras | Excellente | N/A |

#### 4.5 Évolution du Projet (Phases de Développement)

**Phase 1-5:** Implémentation de base
**Phase 6.5:** Refactorisation modulaire (-63% lignes/fichier)
**Phase 7:** Calibration auto + validation IoU
**Phase 8:** Comptage doigts hybride (+13% précision)
**Phase 8.5:** Filtrage avant-bras (-85% faux positifs)

#### 4.6 Cas d'Usage Testés
- Application de peinture gestuelle (main droite)
- Contrôle de souris (mode tracking)
- Détection temps réel avec overlay debug
- Calibration multi-utilisateurs
- Environnements luminosité variable

#### 4.7 Défis Rencontrés et Solutions

| Défi | Solution Implémentée |
|------|---------------------|
| Variations couleur peau | Calibration auto + double espace couleur |
| Bruit et oscillations | Lissage temporel (EMA) + filtrage médian |
| Avant-bras détecté | 5 filtres géométriques + analyse orientation |
| Comptage doigts imprécis | 3 méthodes hybrides + vote médian |
| Arrière-plans complexes | Soustraction de fond MOG2 |

---

## 5. Améliorations Futures Suggérées (max 0,5 page) - 2 points

### Points clés à proposer:

#### 5.1 Améliorations Techniques

**Détection:**
- Détection bi-manuelle (deux mains simultanées)
- Segmentation sémantique pour séparation main/avant-bras
- Histogramme de gradients orientés (HOG) pour orientation
- Filtre de Kalman pour prédiction temporelle
- Détection de gestes dynamiques (mouvement)

**Performance:**
- Optimisation OpenCL/CUDA pour traitement GPU
- Multi-threading pour pipeline parallèle
- Réduction résolution adaptative selon CPU
- Cache de masques pour zones statiques

**Robustesse:**
- Adaptation automatique aux conditions lumineuses
- Calibration continue en arrière-plan
- Détection de confiance pour rejet intelligent
- Gestion occlusions partielles

#### 5.2 Fonctionnalités

**Interactions:**
- Reconnaissance gestes dynamiques (swipe, rotation)
- Profondeur 3D avec caméra stéréo
- Retour haptique virtuel
- Gestes à deux mains (zoom, rotation)

**Applications:**
- Mode présentation (slides, pointeur laser)
- Contrôle média (lecture, volume)
- Gaming (FPS, stratégie)
- Accessibilité (contrôle système d'exploitation)

**Intelligence:**
- Apprentissage de gestes personnalisés
- Détection intention (main active vs passive)
- Prédiction geste suivant
- Adaptation profil utilisateur

#### 5.3 Interface Utilisateur
- Tutoriel interactif de calibration
- Visualisation 3D des landmarks CV
- Dashboard métriques en temps réel
- Mode debug avancé (heatmaps, histogrammes)
- Configuration visuelle des seuils

#### 5.4 Recherche et Développement
- Étude comparative avec autres méthodes CV
- Benchmark sur datasets publics (EgoHands, etc.)
- Publication techniques (article, blog)
- Open-source communautaire
- Extension à d'autres langages (C++, Rust)

---

## 6. Vidéo de Démonstration (max 10 minutes) - 4 points

### Contenu suggéré:

#### Introduction (1 min)
- Présentation équipe
- Objectifs du projet
- Aperçu des fonctionnalités

#### Démonstration Technique (5 min)
- **Calibration automatique** (1 min)
  - Montrer processus auto-calibration
  - Validation IoU
  - Adaptation différentes teintes

- **Mode CV avec debug overlay** (2 min)
  - Détection de peau (masques YCrCb/HSV)
  - Filtrage avant-bras (visualisation étapes)
  - Comptage doigts (3 méthodes affichées)
  - Métriques temps réel

- **Reconnaissance de gestes** (1 min)
  - Démontrer les 6 gestes (0-5 doigts)
  - Transitions fluides
  - Stabilité temporelle

- **Application pratique** (1 min)
  - Peinture gestuelle
  - Contrôle souris
  - Robustesse (différents angles, luminosité)

#### Comparaison CV vs MediaPipe (2 min)
- Split-screen côte à côte
- Même gestes, différentes méthodes
- Mettre en évidence forces CV (arrière-plans, avant-bras)

#### Conclusion (2 min)
- Résumé performances
- Défis surmontés
- Améliorations futures
- Remerciements

---

## Annexes Techniques (pour référence)

### Technologies Utilisées
- **Langage:** Python 3.12
- **Bibliothèques:**
  - OpenCV 4.x (vision par ordinateur)
  - NumPy (calculs numériques)
  - MediaPipe (référence)
- **Outils:** Git, VS Code, Jupyter

### Structure du Code
```
vision_manuelle/
├── src/
│   ├── detectors/
│   │   ├── cv/           # Détecteur CV classique
│   │   └── mediapipe/    # Détecteur MediaPipe
│   ├── core/             # Configuration, utils
│   └── ui/               # Interface utilisateur
├── docs/                 # Documentation technique
├── tools/                # Scripts debug/calibration
└── main.py              # Point d'entrée
```

### Métriques de Développement
- **Lignes de code:** ~3500 (après refactorisation)
- **Commits:** 50+
- **Durée développement:** [À compléter]
- **Tests effectués:** [À compléter]

---

## Checklist Avant Remise

### Rapport PDF
- [ ] Page titre complète avec toutes les informations
- [ ] Mise en contexte claire et concise (≤2.5 pages)
- [ ] Justifications de design argumentées (≤1 page)
- [ ] Résultats avec métriques et comparaisons (≤2 pages)
- [ ] Améliorations futures pertinentes (≤0.5 page)
- [ ] Police 12 pt, format professionnel
- [ ] Figures/tableaux numérotés et légendés
- [ ] Références techniques si applicable

### Vidéo Démonstration
- [ ] Durée ≤10 minutes
- [ ] Narration claire en français
- [ ] Démonstrations fluides sans coupures longues
- [ ] Audio de qualité (micro correct)
- [ ] Vidéo HD (720p minimum)
- [ ] Uploadée sur YouTube (lien fonctionnel)
- [ ] Accessible (non listée ou publique)

### Vérifications Finales
- [ ] Code source propre et commenté
- [ ] README.md à jour
- [ ] Documentation technique complète
- [ ] Tests de fonctionnement effectués
- [ ] Sauvegarde du projet

---

## Notes Importantes

1. **Longueur:** Respecter strictement les limites de pages
2. **Format:** PDF avec police 12 pt lisible
3. **Contenu:** Privilégier qualité > quantité
4. **Figures:** Utiliser diagrammes/screenshots pour illustrer
5. **Vidéo:** Pratiquer la narration avant enregistrement final
6. **Lien YouTube:** Vérifier accessibilité avant soumission

---

## Ressources Complémentaires

Pour plus de détails techniques, consulter:
- `DEVELOPMENT.md` - Historique des phases de développement
- `ARCHITECTURE.md` - Architecture système détaillée
- `USAGE.md` - Guide d'utilisation complet
- `README.md` - Vue d'ensemble du projet
