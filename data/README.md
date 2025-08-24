# 📊 Dataset CIFAR-10

## Téléchargement

Le dataset CIFAR-10 n'est pas inclus dans le repo pour des raisons de taille.

### Option 1: Téléchargement automatique Python
```python
import urllib.request
import tarfile
import os

def download_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    
    # Télécharger
    urllib.request.urlretrieve(url, filename)
    
    # Extraire
    with tarfile.open(filename, 'r:gz') as tar:
        tar.extractall()
    
    # Déplacer les fichiers
    import shutil
    for file in os.listdir('cifar-10-batches-py'):
        shutil.move(f'cifar-10-batches-py/{file}', f'data/{file}')
    
    # Nettoyer
    os.remove(filename)
    shutil.rmtree('cifar-10-batches-py')

if __name__ == "__main__":
    download_cifar10()
```

### Option 2: Téléchargement manuel
1. Aller sur https://www.cs.toronto.edu/~kriz/cifar.html
2. Télécharger "CIFAR-10 python version"
3. Extraire dans le dossier `data/`

## Structure attendue
```
data/
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── batches.meta
```

## Format des données
- **Images** : 32x32 pixels, 3 canaux (RGB)
- **Classes** : 10 classes (avion, auto, oiseau, chat, cerf, chien, grenouille, cheval, bateau, camion)
- **Train** : 50,000 images (5 batches de 10,000)
- **Test** : 10,000 images
