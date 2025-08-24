import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def create_dataloaders(data_path,batch_size=128):
    
    (train_data, train_labels), (test_data, test_labels) = cifar10_data(data_path)
    
    # Normalisation
    train_data = torch.FloatTensor(train_data) / 255.0 
    test_data = torch.FloatTensor(test_data) / 255.0
    
    # Permutation des dimensions pour PyTorch (H, W, C) -> (C, H, W)
    train_data = train_data.permute(0, 3, 1, 2)
    test_data = test_data.permute(0, 3, 1, 2)
    
    # Créer le dataset et dataloader
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    
    # Créer le dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def create_test_dataloader(test_batch_path, batch_size=128):
    """Crée un dataloader uniquement pour le fichier test_batch"""
    
    # Charger uniquement le test_batch
    test_data, test_labels = load_data(test_batch_path)
    
    # Normalisation
    test_data = torch.FloatTensor(test_data) / 255.0
    
    # Permutation des dimensions pour PyTorch (H, W, C) -> (C, H, W)
    test_data = test_data.permute(0, 3, 1, 2)
    
    # Créer le dataset et dataloader
    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader

def load_data(file):
    batch = unpickle(file)
    data = batch[b'data']
    labels = batch[b'labels']

    data = data.reshape((data.shape[0], 3, 32, 32))
    data = data.transpose((0, 2, 3, 1)) # (10000, 3, 32, 32) to (10000, 32, 32, 3)      
    labels = torch.tensor(labels)

    return data, labels

def cifar10_data(data_path):
    train_data = []
    train_labels = []
    
    for i in range (1, 6):
        batch_path = os.path.join(data_path, f'data_batch_{i}')
        data, labels = load_data(batch_path)
        train_data.append(data)
        train_labels.append(labels)
    
    train_data = np.vstack(train_data)
    train_labels = torch.cat(train_labels)
    test_path = os.path.join(data_path, 'test_batch')
    test_data, test_labels = load_data(test_path)
    return (train_data, train_labels),(test_data,test_labels)


    
        

