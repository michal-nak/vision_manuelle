# Détecteur de main (classique, sans apprentissage)

Ce dépôt contient un démonstrateur de détection de main construit avec des méthodes classiques de vision par ordinateur (pas de modèle d'apprentissage).

Caractéristiques
- Segmentation par couleur peau en espace YCrCb
- Nettoyage par opérations morphologiques
- Extraction du plus grand contour comme main
- Calcul du convex hull et des convexity defects pour estimer les points des doigts
- Support pour la caméra en direct (webcam)

Prérequis
- Python 3.10+
- Installer les dépendances listées dans `requirements.txt` :

```bash
pip install -r requirements.txt
```

Usage

Lancer la démo webcam :

```bash
python3 hand_detection.py --camera 0
```

Touches utiles
- `q` : quitter
- `c` : capturer et sauvegarder l'image annotée (capture_<n>.png)

Notes
- Les paramètres de seuils pour la segmentation peau peuvent nécessiter un ajustement selon l'éclairage et le ton de peau.
- Cette approche est simple et rapide, mais moins robuste que des méthodes basées sur des modèles (ex. MediaPipe Hand, réseaux CNN).

Si tu veux, je peux :
- ajouter des paramètres CLI pour ajuster les seuils en temps réel
- fournir une version supportant l'enregistrement vidéo
- intégrer MediaPipe (si tu veux un détecteur beaucoup plus robuste)
# vision_manuelle
c'est pour voir manu

## Proposition de Projet

### 1. Contenu de l'équipe 7 :

<ul>
<li> Edward Leroux </li>
<li> Michal Naumiak </li>
<li> François Gerbeau </li>
<li> Théo Lahmar </li>
</ul>

### 2. Choix du sujet :
**Contrôle par la main d’un logiciel graphique**


### 3. Description du projet et de son lien avec la matière du cours:

Ce système permettrait de contrôler l’interface d’un logiciel graphique type Paint au moyen de gestes de la main devant une caméra. Le geste définirait le choix d’un outil puis son utilisation. Le geste est observé au moyen d’une caméra de type webcam. Le projet implique une partie de détection/segmentation de la main, du suivi dans une séquence vidéo de celui-ci et de l’encodage des gestes en vue d’en faire la reconnaissance ainsi qu’une partie de traitement de l’image pour un rendu optimisé.

### 4. Une liste des équipements et logiciels requis avec leur disponibilité vérifiée:

<ul>
<li> Un ordinateur portable avec une webcam (disponibilité assurée par l’un au moins des membres du groupe) </li>
<li> Le logiciel graphique open source choisi installé sur l’ordinateur muni de la webcam </li>
</ul>

