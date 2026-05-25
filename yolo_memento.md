# 🧠 Mémento — Paramètres d'entraînement YOLO (Ultralytics)

> Référence synthétique pour comprendre, ajuster et tuner un entraînement YOLO.

---

## 📁 Données & Infrastructure

| Paramètre | Valeur exemple | Rôle | Quand modifier |
|-----------|---------------|------|----------------|
| `data` | `'data.yaml'` | Chemin vers le fichier de config du dataset (classes, chemins train/val) | Toujours à définir |
| `device` | `0` | GPU à utiliser (`0`, `1`, `"cpu"`, `"0,1"` pour multi-GPU) | Si plusieurs GPU ou CPU forcé |
| `workers` | `2` | Threads de chargement des données | ↑ si I/O lente et RAM dispo, ↓ si crash/OOM |
| `cache` | `"ram"` | Met le dataset en cache RAM (`"ram"`), disque (`"disk"`), ou désactivé (`False`) | `"ram"` si dataset petit, `"disk"` sinon, `False` si contrainte mémoire |
| `project` | `'runs'` | Dossier de sortie des résultats | Organiser plusieurs expériences |

---

## 🔁 Boucle d'entraînement

| Paramètre | Valeur exemple | Rôle | Quand modifier |
|-----------|---------------|------|----------------|
| `epochs` | `350` | Nombre total de passes sur le dataset | ↑ si underfitting, ↓ si overfitting rapide (combiné avec `patience`) |
| `batch` | `24` | Images par batch | ↓ si OOM GPU, ↑ pour stabiliser les gradients (si VRAM le permet) |
| `imgsz` | `640` | Taille de redimensionnement des images (px) | ↑ pour détecter des petits objets (coût mémoire ×4 si ×2 résolution) |
| `patience` | `50` | Early stopping : arrêt si pas d'amélioration depuis N epochs | ↓ pour stopper plus tôt, ↑ si entraînement bruité |

---

## ⚙️ Optimiseur & Hyperparamètres

| Paramètre | Valeur exemple | Rôle | Quand modifier |
|-----------|---------------|------|----------------|
| `optimizer` | `"MuSGD"` | Algorithme d'optimisation (`SGD`, `Adam`, `AdamW`, `MuSGD`…) | `SGD`/`MuSGD` = plus robuste en généralisation ; `Adam` = converge plus vite |
| `momentum` | `0.937` | Mémoire du gradient précédent (SGD) | Valeur standard ~0.9–0.95, rarement à changer |
| `weight_decay` | `0.0005` | Régularisation L2 — pénalise les gros poids | ↑ si overfitting, ↓ si underfitting |
| `dropout` | `0.1` | Désactive aléatoirement X% des neurones (régularisation) | ↑ si overfitting (max ~0.3–0.5), `0.0` pour désactiver |

---

## 🔥 Warmup

| Paramètre | Valeur exemple | Rôle | Quand modifier |
|-----------|---------------|------|----------------|
| `warmup_epochs` | `5` | Montée progressive du learning rate au démarrage | ↑ si instabilité en début d'entraînement, `0` si LR stable dès le départ |

> **Pourquoi ?** Un LR trop fort dès l'epoch 0 peut faire diverger le modèle. Le warmup évite ce crash initial.

---

## 🎨 Data Augmentation

Ces paramètres modifient aléatoirement les images à la volée pendant l'entraînement pour **forcer le modèle à généraliser**.

### Géométrique

| Paramètre | Valeur exemple | Effet | Conseil |
|-----------|---------------|-------|---------|
| `fliplr` | `0.25` | Probabilité de miroir horizontal | `0.5` standard ; ↓ si objets asymétriques (ex : panneaux directionnels) |
| `flipud` | `0.1` | Probabilité de miroir vertical | `0.0` si objets ont un sens haut/bas (personnes, voitures…) |
| `mosaic` | `0.0` | Fusionne 4 images en une — excellent pour la détection multi-échelle | `1.0` recommandé pour la plupart des cas ; `0.0` si dataset déjà très varié ou petits objets précis |

### Couleur (HSV)

| Paramètre | Valeur exemple | Effet | Conseil |
|-----------|---------------|-------|---------|
| `hsv_h` | `0.015` | Variation de teinte (hue) | ↑ si couleurs très variables dans la réalité |
| `hsv_s` | `0.35` | Variation de saturation | ↑ pour robustesse aux conditions lumineuses |
| `hsv_v` | `0.2` | Variation de luminosité (value) | ↑ pour robustesse jour/nuit ou éclairage variable |

---

## 🧭 Paramètres à prioriser selon le problème

| Situation | Action suggérée |
|-----------|----------------|
| **Overfitting** (val loss monte, train loss descend) | ↑ `weight_decay`, ↑ `dropout`, ↑ augmentations, ↓ `epochs` |
| **Underfitting** (les deux losses stagnent) | ↑ `epochs`, ↑ `imgsz`, ↓ `weight_decay`, ↓ `dropout` |
| **Mémoire GPU insuffisante** | ↓ `batch`, ↓ `imgsz`, `cache=False` |
| **Entraînement lent** | ↑ `workers`, `cache="ram"`, ↑ `batch` |
| **Mauvaise détection de petits objets** | ↑ `imgsz`, ↑ `mosaic`, ↓ `hsv_v` |
| **Généralisation sur conditions variées** | ↑ augmentations HSV, ↑ `fliplr`, activer `mosaic` |

---

## 📊 Lire les logs d'entraînement

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/350      1.25G     0.7947      7.887   0.009755          3        640

           Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
             all         61         96    0.00757      0.662     0.0548     0.0391
```

### Ligne de train (par batch)

| Colonne | Signification | Tendance souhaitée |
|---------|--------------|-------------------|
| `Epoch` | Epoch en cours / total | — |
| `GPU_mem` | VRAM utilisée | Stable, pas trop proche du max |
| `box_loss` | Erreur sur la position/taille des boîtes | ↓ le plus possible |
| `cls_loss` | Erreur de classification (mauvaise classe) | ↓ le plus possible |
| `dfl_loss` | Erreur sur la distribution des bords de boîte (précision fine) | ↓ le plus possible |
| `Instances` | Nb d'objets dans le batch courant | Info seulement |
| `Size` | Taille d'image utilisée | Info seulement |

> Les 3 losses doivent **descendre ensemble**. Si `cls_loss` reste haute → le modèle confond les classes. Si `box_loss` stagne → il localise mal.

### Ligne de validation (métriques)

| Colonne | Signification | Cible |
|---------|--------------|-------|
| `Class` | Classe évaluée (`all` = moyenne) | — |
| `Images` | Nb d'images dans le set de validation | — |
| `Instances` | Nb total d'objets annotés dans la val | — |
| `Box(P)` | **Précision** — parmi les détections, combien sont correctes ? | → 1.0 |
| `R` | **Rappel** — parmi les vrais objets, combien sont détectés ? | → 1.0 |
| `mAP50` | Mean Average Precision à seuil IoU 0.5 — métrique principale | → 1.0 |
| `mAP50-95` | mAP moyenné sur IoU 0.5 à 0.95 — plus exigeant sur la précision des boîtes | → 1.0 |

### Le compromis Précision / Rappel

- **Précision haute, rappel bas** → le modèle est prudent, il détecte peu mais se trompe rarement. Il *rate* des objets.
- **Rappel haut, précision bas** → le modèle détecte tout, mais génère beaucoup de faux positifs.
- **L'idéal** : les deux proches de 1. En pratique, on cherche à maximiser `mAP50` qui les équilibre.

### mAP50 vs mAP50-95

| Métrique | Ce qu'elle mesure | Quand la prioriser |
|----------|------------------|-------------------|
| `mAP50` | Détection correcte si la boîte chevauche à ≥50% | Cas standard, objets larges |
| `mAP50-95` | Idem mais boîte très précise requise (jusqu'à 95% de chevauchement) | Si la précision spatiale compte (robotique, mesure…) |

> **Ton epoch 1** : `mAP50 = 0.055` et `R = 0.66` → le modèle trouve déjà des objets (rappel ok) mais ses boîtes et classifications sont encore mauvaises. C'est tout à fait normal au début.

---

## 💡 Valeurs par défaut YOLO utiles à connaître

| Paramètre | Défaut YOLO | Note |
|-----------|------------|------|
| `lr0` | `0.01` | Learning rate initial (non défini dans ton script → valeur auto) |
| `lrf` | `0.01` | LR final en fraction du LR initial |
| `mosaic` | `1.0` | Désactivé dans ton script (`0.0`) — volontaire ? |
| `fliplr` | `0.5` | Ton script utilise `0.25` — plus conservateur |
| `warmup_epochs` | `3.0` | Ton script monte à `5` — plus prudent |

---

*Généré comme aide-mémoire personnel — basé sur la doc Ultralytics YOLO v8/v11.*
