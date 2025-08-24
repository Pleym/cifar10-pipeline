# 🚀 MLOps Pipeline CIFAR-10

Pipeline MLOps complet pour la classification d'images CIFAR-10 avec PyTorch et MLflow tracking.

## 📋 Fonctionnalités

- **🧠 Modèle CNN configurable** : Architecture dynamique via YAML
- **📊 MLflow tracking** : Suivi des expériences et métriques
- **📈 Visualisations avancées** : Matrice de confusion, accuracy par classe
- **🔧 Configuration centralisée** : Hyperparamètres dans `configs/model.yaml`
- **🧪 Pipeline de test** : Évaluation complète avec graphiques

## 🏗️ Architecture

```
Pipeline_ML/
├── configs/
│   └── model.yaml          # Configuration des hyperparamètres
├── src/
│   ├── data/
│   │   └── dataset.py       # Chargement et preprocessing CIFAR-10
│   ├── models/
│   │   └── model.py         # Architecture CNN
│   └── utils/
│       └── config.py        # Gestion de la configuration
├── scripts/
│   ├── train.py            # Script d'entraînement
│   └── test.py             # Script d'évaluation avec graphiques
└── data/                   # Dataset CIFAR-10 (non versionné)
```

## 🚀 Installation

### Prérequis
- Python 3.11+
- UV (gestionnaire de dépendances)

### Setup
```bash
# Cloner le repo
git clone <your-repo-url>
cd Pipeline_ML

# Installer les dépendances
uv sync

# Télécharger CIFAR-10 dans le dossier data/
# (Instructions dans data/README.md)
```

## 📊 Utilisation

### Entraînement
```bash
uv run scripts/train.py
```

### Évaluation
```bash
uv run scripts/test.py
```

### MLflow UI
```bash
mlflow ui
# Ouvrir http://localhost:5000
```

## ⚙️ Configuration

Modifier `configs/model.yaml` pour ajuster :
- Architecture du modèle (couches conv/FC)
- Hyperparamètres d'entraînement
- Paramètres de logging

## 📈 Résultats

- **Accuracy** : ~73% sur CIFAR-10 test set
- **Architecture** : 3 couches conv + 3 couches FC
- **Optimiseur** : Adam avec learning rate 0.001

## 🔧 Technologies

- **PyTorch** : Framework deep learning
- **MLflow** : Tracking des expériences
- **UV** : Gestion des dépendances
- **Matplotlib/Seaborn** : Visualisations
- **scikit-learn** : Métriques d'évaluation

## 📝 TODO - Roadmap

### Phase 2 - HPC
- [ ] Parallélisation DistributedDataParallel
- [ ] Support multi-GPU
- [ ] Optimisations HPC

### Phase 3 - MLOps Avancé
- [ ] Tests unitaires
- [ ] CI/CD Pipeline
- [ ] Containerisation Docker
- [ ] Déploiement Kubernetes

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature
3. Commit les changements
4. Push vers la branche
5. Ouvrir une Pull Request

## 📄 License

MIT License - voir LICENSE file pour détails.