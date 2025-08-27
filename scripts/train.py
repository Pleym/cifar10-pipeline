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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Sélection du modèle (par nom) sans casser l'ancien comportement
    model_cfg = dict(config.model.__dict__)
    model_name = model_cfg.pop("name", "DeepCNN")

    if model_name == "SimpleCNN":
        model_cls = SimpleCNN
    elif model_name == "DeepCNN":
        # Import tardif; essaie d'abord src.models, puis src.model_zoo
        try:
            from src.models.deep_cnn import DeepCNN 
            model_cls = DeepCNN
        except Exception:
            try:
                from src.model_zoo.deep_cnn import DeepCNN  
                model_cls = DeepCNN
            except Exception as e:
                raise ImportError(
                    "DeepCNN selected but not found. "
                    "or switch model.name back to 'SimpleCNN'."
                ) from e
    else:
        raise ValueError(f"Unknown model.name '{model_name}'. Use 'SimpleCNN' or 'DeepCNN'.")

    # Créer le modèle
    model = model_cls(**model_cfg).to(device)
    
    # Charger les données
    train_loader, test_loader = create_dataloaders(
        config.data.data_path, 
        config.training.batch_size
    )
    
    # Optimiseur et loss
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # MLflow tracking
    if model_name == "SimpleCNN":
        mlflow.set_experiment(config.logging.experiment_name + "_simple")
    elif model_name == "DeepCNN":
        mlflow.set_experiment(config.logging.experiment_name + "_deep")
    with mlflow.start_run():
        # Logger les hyperparamètres
        mlflow.log_params(config.training.__dict__)
        mlflow.log_params(config.model.__dict__)
        
        # Boucle d'entraînement
        def topk_acc(logits: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
            if logits.size(1) < k:
                return float('nan')
            _, pred = logits.topk(k, dim=1)
            correct = pred.eq(targets.view(-1, 1)).any(dim=1).float().mean().item()
            return correct

        for epoch in range(config.training.num_epochs):
            model.train()
            running_loss = 0.0
            running_correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                # Stats
                batch_size = data.size(0)
                running_loss += loss.item() * batch_size
                running_correct += (output.argmax(dim=1) == target).sum().item()
                total += batch_size

                # Log per-step loss
                if batch_idx % config.logging.log_interval == 0:
                    step = epoch * len(train_loader) + batch_idx
                    mlflow.log_metric("train_loss", loss.item(), step=step)

            # Epoch-level train metrics
            epoch_train_loss = running_loss / max(total, 1)
            epoch_train_acc = running_correct / max(total, 1)
            mlflow.log_metric("epoch_train_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("epoch_train_acc", epoch_train_acc, step=epoch)

            # Log learning rate (first param group)
            current_lr = optimizer.param_groups[0]["lr"]
            mlflow.log_metric("lr", current_lr, step=epoch)

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_top5_acc = []
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device)
                    target = target.to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item() * data.size(0)
                    val_correct += (output.argmax(dim=1) == target).sum().item()
                    val_total += data.size(0)
                    # top-5 if applicable
                    acc5 = topk_acc(output, target, k=5)
                    if acc5 == acc5:  # not NaN
                        val_top5_acc.append(acc5)

            epoch_val_loss = val_loss / max(val_total, 1)
            epoch_val_acc = val_correct / max(val_total, 1)
            mlflow.log_metric("val_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("val_acc", epoch_val_acc, step=epoch)
            if len(val_top5_acc) > 0:
                mlflow.log_metric("val_top5_acc", sum(val_top5_acc) / len(val_top5_acc), step=epoch)

            print(
                f"Epoch {epoch+1}/{config.training.num_epochs} | "
                f"train_loss: {epoch_train_loss:.4f}, train_acc: {epoch_train_acc:.4f} | "
                f"val_loss: {epoch_val_loss:.4f}, val_acc: {epoch_val_acc:.4f}"
            )

        # Sauvegarder le modèle
        if model_name == "SimpleCNN":
            mlflow.pytorch.log_model(model, "model_simple")
        elif model_name == "DeepCNN":
            mlflow.pytorch.log_model(model, "model_deep")
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