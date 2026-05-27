
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
            tranforms.RandomRotation(degrees=15),          
            
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
    
     
    