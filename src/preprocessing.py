
import os
import numpy as np
import cv2
from tqdm import tqdm


def load_dataset(dataset_path, img_classes):
    
    """
    Carica tutte le immagini dal dataset PlantVillage.

    Parametri:
        dataset_path (str): path alla cartella radice del dataset
        classi (list):      lista dei nomi delle sottocartelle/classi
                            es. ["Tomato_healthy", "Tomato_Early_blight", ...]

    Ritorna:
        X (np.ndarray): array di immagini, shape (N, 224, 224, 3), dtype uint8, RGB
        y (np.ndarray): array di label numeriche, shape (N,), dtype int
    """
    
    x = []
    y = []
    
    for label, img_class in enumerate(img_classes):
        
        class_path = os.path.join(dataset_path, img_class)
        
        if not os.path.exists(class_path):
            print(f"Attenzione: {class_path} non esiste. Saltando.")
            continue
        if not os.path.isdir(class_path):
            print(f"Attenzione: {class_path} non è una cartella. Saltando.")
            continue
        
        file_list = os.listdir(class_path)
        
        for filename in tqdm(file_list, desc=f"Caricamento {img_class}"):
            
            img_path = os.path.join(class_path, filename)
            
            try: 
                
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Attenzione: {img_path} non è un'immagine valida. Saltando.")
                    continue
                
                img = cv2.resize(img, (224, 224))
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                x.append(img)
                y.append(label)
            
            except Exception as e:
                print(f"Errore caricando {img_path}: {e}. Saltando.")
                continue
            
    X = np.array(x, dtype=np.uint8)
    y = np.array(y, dtype=np.int64)
    
    
    print(f"\n✅ Dataset caricato!")
    print(f"   Totale immagini : {len(X)}")
    print(f"   Shape X         : {X.shape}")
    print(f"   Shape y         : {y.shape}")
    print(f"   Distribuzione   : {np.bincount(y)}")
    
    return X, y
    
    

from torchvision import transforms


def get_transforms():
    
    """
    Definisce le trasformazioni PyTorch per train e val/test.

    - Train: augmentation + normalizzazione ImageNet
    - Val/Test: solo resize + normalizzazione (niente augmentation)

    Come da Modulo 3: l'augmentation è una tecnica di regolarizzazione
    che previene l'overfitting aumentando artificialmente la varietà
    del training set.

    Ritorna:
        dict con chiavi 'train' e 'val', valori Compose di trasformazioni
    """
    
    mean = [0.485, 0.456, 0.406]  # media ImageNet
    std = [0.229, 0.224, 0.225]   # std ImageNet
    
    transform_train = transforms.Compose([
        
            # Resize a 224x224 (anche se già fatto in load_dataset, è buona pratica includerlo qui)
            transforms.Resize((224, 224)),

            # Augmentation: flip orizzontale con 50% di probabilità
            transforms.RandomHorizontalFlip(p=0.5),
            
            # Rotazione casuale entro ±15 gradi
            transforms.RandomRotation(degrees=15),          
            
            # Varia luminosità e contrasto
            # Simula condizioni di luce diverse — aiuta contro il bias
            # del laboratorio di PlantVillage (sfondo uniforme, luce fissa)
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            
            #Converti in tensore
            transforms.ToTensor(), 
            
            # Normalizzazione con media e std di ImageNet
            transforms.Normalize(mean=mean, std=std)
        
        ])
    
    
    transform_val = transforms.Compose([
        
        transforms.Resize((224, 224)),
        
        transforms.ToTensor(),
        
        transforms.Normalize(mean=mean, std=std)
        
    ])
    
    
    return {
        'train': transform_train,
        'val': transform_val,
        'test': transform_val  # stessa trasformazione di val per test
    }
    
     

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import torch

# =============================================================
# Classe Dataset personalizzata per PlantVillage
# PyTorch richiede una classe Dataset con __len__ e __getitem__
# =============================================================

class PlantVillageDataset(Dataset):
    """
    Dataset PyTorch personalizzato per le immagini PlantVillage.
    Applica le trasformazioni on-the-fly ad ogni immagine richiesta.
    """

    def __init__(self, X, y, transform=None):
        """
        Parametri:
            X (np.ndarray): immagini shape (N, 224, 224, 3) uint8 RGB
            y (np.ndarray): label numeriche shape (N,)
            transform: trasformazioni torchvision da applicare
        """
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        # PyTorch chiama questo metodo per sapere quante immagini ci sono
        return len(self.X)

    def __getitem__(self, idx):
        # PyTorch chiama questo metodo per ottenere l'immagine numero idx

        # Prende l'immagine NumPy uint8 RGB
        img = self.X[idx]

        # Converte in PIL Image — richiesto da torchvision transforms
        img = Image.fromarray(img)

        # Applica le trasformazioni (ToTensor + Normalize + augmentation)
        if self.transform:
            img = self.transform(img)

        # Converte la label in tensore
        label = torch.tensor(self.y[idx], dtype=torch.long)

        return img, label


# =============================================================
# Funzione principale get_dataloaders
# =============================================================

def get_dataloaders(X, y, transforms_dict, batch_size=32, random_state=42):
    """
    Divide il dataset in train/val/test e crea i DataLoader PyTorch.

    Split: 80% train, 10% val, 10% test
    random_state fisso per riproducibilità — stessa divisione ad ogni run.

    Parametri:
        X (np.ndarray)      : immagini (N, 224, 224, 3)
        y (np.ndarray)      : label (N,)
        transforms_dict     : dizionario da get_transforms()
        batch_size (int)    : dimensione batch, default 32
        random_state (int)  : seed per riproducibilità, default 42

    Ritorna:
        train_loader, val_loader, test_loader
    """

    # ----------------------------------------------------------
    # STEP 1 — Split 80/10/10
    # Prima splittiamo 80/20, poi il 20% lo dividiamo a metà
    # ----------------------------------------------------------

    # Split principale: 80% train, 20% temporaneo
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.2,          # 20% va in temp
        random_state=random_state,
        stratify=y              # mantiene le proporzioni delle classi
    )

    # Split del temporaneo: 50% val, 50% test → 10% e 10% del totale
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,          # metà va in test
        random_state=random_state,
        stratify=y_temp         # mantiene le proporzioni delle classi
    )

    # Riepilogo dello split
    print(f"✅ Split completato:")
    print(f"   Train : {len(X_train)} immagini ({len(X_train)/len(X)*100:.0f}%)")
    print(f"   Val   : {len(X_val)} immagini ({len(X_val)/len(X)*100:.0f}%)")
    print(f"   Test  : {len(X_test)} immagini ({len(X_test)/len(X)*100:.0f}%)")

    # ----------------------------------------------------------
    # STEP 2 — Crea i Dataset PyTorch
    # Ogni split riceve la trasformazione corretta
    # ----------------------------------------------------------

    train_dataset = PlantVillageDataset(X_train, y_train, transform=transforms_dict['train'])
    val_dataset   = PlantVillageDataset(X_val,   y_val,   transform=transforms_dict['val'])
    test_dataset  = PlantVillageDataset(X_test,  y_test,  transform=transforms_dict['test'])

    # ----------------------------------------------------------
    # STEP 3 — Crea i DataLoader
    # ----------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # mescola ad ogni epoca — fondamentale per il training
        num_workers=2       # carica i dati in parallelo per velocità
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # nessun mescolamento — stiamo solo misurando
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,      # nessun mescolamento — valutazione finale
        num_workers=2
    )

    return train_loader, val_loader, test_loader