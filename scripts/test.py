import torch
import torch.nn as nn
import sys
import os
import mlflow
import mlflow.pytorch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import tempfile
# Ajouter le répertoire parent au path pour pouvoir importer src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import create_test_dataloader
from src.utils.config import load_config


def test_model():
    config = load_config("configs/model.yaml")
    
    
    experiment = mlflow.get_experiment_by_name(config.logging.experiment_name)
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    latest_run_id = runs.iloc[0]['run_id']
    model = mlflow.pytorch.load_model(f"runs:/{latest_run_id}/model")
    model.eval()
    test_loader = create_test_dataloader("data/test_batch", config.training.batch_size)
    
    # Classes CIFAR-10
    classes = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 
               'chien', 'grenouille', 'cheval', 'bateau', 'camion']
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Stockage pour les graphiques
    all_predictions = []
    all_targets = []
    batch_losses = []
    batch_accuracies = []
    
    mlflow.set_experiment(config.logging.experiment_name + "_test")
    with mlflow.start_run():
        mlflow.log_params(config.model.__dict__)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculer l'accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Stocker pour les graphiques
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                batch_losses.append(loss.item())
                
                # Accuracy par batch
                batch_acc = (predicted == targets).sum().item() / targets.size(0) * 100
                batch_accuracies.append(batch_acc)
                
                # Log métriques par batch
                mlflow.log_metric("batch_test_loss", loss.item(), step=batch_idx)
                mlflow.log_metric("batch_test_accuracy", batch_acc, step=batch_idx)
        
        # Métriques finales
        avg_loss = total_loss / len(test_loader)
        accuracy = 100 * correct / total
        
        mlflow.log_metric("test_loss", avg_loss)
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Créer et logger les graphiques
        create_evaluation_plots(all_predictions, all_targets, batch_losses, batch_accuracies, classes)
        
        print(f"📊 Résultats du test:")
        print(f"   Loss moyenne: {avg_loss:.4f}")
        print(f"   Accuracy: {accuracy:.2f}% ({correct}/{total})")

def create_evaluation_plots(predictions, targets, batch_losses, batch_accuracies, classes):
    """Crée et sauvegarde les graphiques d'évaluation dans MLflow"""
    
    # 1. Matrice de confusion
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de Confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Classes')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(tmp.name, "plots/confusion_matrix.png")
    plt.close()
    
    # 2. Accuracy par classe
    plt.figure(figsize=(12, 6))
    class_accuracies = []
    for i, class_name in enumerate(classes):
        class_mask = np.array(targets) == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(np.array(predictions)[class_mask] == i) * 100
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0)
    
    bars = plt.bar(classes, class_accuracies, color='skyblue', edgecolor='navy')
    plt.title('Accuracy par Classe')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars, class_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom')
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(tmp.name, "plots/class_accuracy.png")
    plt.close()
    
    # 3. Évolution de la loss par batch
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(batch_losses, color='red', alpha=0.7)
    plt.title('Loss par Batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # 4. Évolution de l'accuracy par batch
    plt.subplot(1, 2, 2)
    plt.plot(batch_accuracies, color='green', alpha=0.7)
    plt.title('Accuracy par Batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(tmp.name, "plots/batch_metrics.png")
    plt.close()
    
    # 5. Distribution des prédictions
    plt.figure(figsize=(10, 6))
    unique_preds, counts_preds = np.unique(predictions, return_counts=True)
    unique_targets, counts_targets = np.unique(targets, return_counts=True)
    
    x = np.arange(len(classes))
    width = 0.35
    
    plt.bar(x - width/2, [counts_targets[i] if i in unique_targets else 0 for i in range(len(classes))], 
            width, label='Vraies Classes', alpha=0.8, color='lightblue')
    plt.bar(x + width/2, [counts_preds[i] if i in unique_preds else 0 for i in range(len(classes))], 
            width, label='Prédictions', alpha=0.8, color='orange')
    
    plt.title('Distribution des Classes: Vraies vs Prédites')
    plt.xlabel('Classes')
    plt.ylabel('Nombre d\'échantillons')
    plt.xticks(x, classes, rotation=45)
    plt.legend()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(tmp.name, "plots/class_distribution.png")
    plt.close()

if __name__ == "__main__":
    test_model()
    
    


    