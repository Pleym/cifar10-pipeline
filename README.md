# 🚀 MLOps Pipeline CIFAR-10

 Complete MLOps pipeline for CIFAR-10 image classification with PyTorch and MLflow tracking.

## 📋 Features

- 🧠 Configurable CNN model (YAML-driven architecture)
- 📊 MLflow tracking for experiments and metrics
- 📈 Advanced visualizations: confusion matrix, per-class accuracy, batch curves
- 🔧 Centralized configuration in `configs/model.yaml`
- 🧪 Testing pipeline with automatic plots logged to MLflow

## 🏗️ Project Structure

```
Pipeline_ML/
├── configs/
│   └── model.yaml          # Hyperparameter configuration
├── src/
│   ├── data/
│   │   └── dataset.py      # CIFAR-10 loading & preprocessing
│   ├── models/
│   │   └── model.py        # CNN architecture
│   └── utils/
│       └── config.py       # Config management
├── scripts/
│   ├── train.py            # Training script
│   └── test.py             # Evaluation + plots
└── data/                   # CIFAR-10 dataset (not versioned)
```

## 🚀 Installation

### Prerequisites
- Python 3.11+
- UV (dependency manager)

### Setup
```bash
# Clone the repo
git clone https://github.com/Pleym/cifar10-pipeline.git
cd cifar10-pipeline

# Install dependencies
uv sync

# Download CIFAR-10 into the data/ folder
# (see data/README.md for instructions)
```

## 📊 Usage

### Training
```bash
uv run scripts/train.py
```

### Evaluation
```bash
uv run scripts/test.py
```

### MLflow UI
```bash
mlflow ui
# Open http://localhost:5000
```

## ⚙️ Configuration

Edit `configs/model.yaml` to adjust:
- Model architecture (conv/FC layers)
- Training hyperparameters
- Logging parameters

## 📈 Results
You can observe the result with the MLflow UI.
- Accuracy: ~73% on CIFAR-10 test set
- Architecture: 3 convolution layers + 3 fully-connected layers
- Optimizer: Adam, learning rate 0.001

## 🔧 Tech Stack

- PyTorch — deep learning
- MLflow — experiment tracking
- UV — dependency management
- Matplotlib/Seaborn — visualization
- scikit-learn — evaluation metrics

## 📝 Roadmap

### Phase 2 — HPC
- [ ] DistributedDataParallel (DDP)
- [ ] Multi-GPU support
- [ ] HPC optimizations
- [ ] Unit tests
- [ ] CI/CD pipeline
- [ ] Docker containerization
- [ ] Kubernetes deployment

## 📄 License

MIT License — see LICENSE file for details.