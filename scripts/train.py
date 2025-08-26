import torch
import torch.nn as nn
import argparse
import sys
import os
import mlflow
import mlflow.pytorch

# Ajouter le répertoire parent au path pour pouvoir importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.model import SimpleCNN
from src.data.dataset import create_dataloaders
from src.utils.config import load_config

def train_model(config_path: str = "configs/model.yaml"):
    # Charger la configuration
    config = load_config(config_path)
    
    # Sélection du modèle (par nom) sans casser l'ancien comportement
    model_cfg = dict(config.model.__dict__)
    model_name = model_cfg.pop("name", "SimpleCNN")

    if model_name == "SimpleCNN":
        model_cls = SimpleCNN
    elif model_name == "DeepCNN":
        # Import tardif; essaie d'abord src.models, puis src.model_zoo
        try:
            from src.models.deep_cnn import DeepCNN  # type: ignore
            model_cls = DeepCNN
        except Exception:
            try:
                from src.model_zoo.deep_cnn import DeepCNN  # type: ignore
                model_cls = DeepCNN
            except Exception as e:
                raise ImportError(
                    "DeepCNN selected but not found. Provide src/models/deep_cnn.py or src/model_zoo/deep_cnn.py, "
                    "or switch model.name back to 'SimpleCNN'."
                ) from e
    else:
        raise ValueError(f"Unknown model.name '{model_name}'. Use 'SimpleCNN' or 'DeepCNN'.")

    # Créer le modèle
    model = model_cls(**model_cfg)
    
    # Charger les données
    train_loader, test_loader = create_dataloaders(
        config.data.data_path, 
        config.training.batch_size
    )
    
    # Optimiseur et loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # MLflow tracking
    mlflow.set_experiment(config.logging.experiment_name)
    with mlflow.start_run():
        # Logger les hyperparamètres
        mlflow.log_params(config.training.__dict__)
        mlflow.log_params(config.model.__dict__)
        
        # Boucle d'entraînement
        for epoch in range(config.training.num_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Logger la loss périodiquement
                if batch_idx % config.logging.log_interval == 0:
                    step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metric("train_loss", loss.item(), step=step)
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
                
        
        # Sauvegarder le modèle
        mlflow.pytorch.log_model(model, "model")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CIFAR-10 model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    train_model(args.config)