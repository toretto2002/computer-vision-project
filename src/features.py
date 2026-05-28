"""
Modulo 2 - Feature Extraction Classica: HOG (Histograms of Oriented Gradients)
Come visto a lezione, HOG cattura la forma degli oggetti attraverso la
distribuzione dei gradienti locali in celle e blocchi normalizzati.
"""

import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm  # barra di progresso


# ─────────────────────────────────────────────
# COSTANTI HOG (parametri da Modulo 2)
# ─────────────────────────────────────────────
# Dimensione di resize per HOG (più piccola di 224 → più veloce)
HOG_IMG_SIZE = (128, 128)

# Parametri HOG coerenti con quanto visto a lezione:
# - 9 orientazioni: cattura gradienti da 0° a 180°
# - celle 8×8 pixel: granularità locale fine
# - blocchi 2×2 celle: normalizzazione per robustezza all'illuminazione
HOG_ORIENTATIONS   = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)

# Mapping classi → label numeriche (coerente con preprocessing.py)
CLASSI = {
    "Tomato_healthy":     0,
    "Tomato_Early_blight": 1,
    "Tomato_Late_blight":  2,
}

def extract_hog_features(img_bgr: np.ndarray) -> np.ndarray:
    """
    Estrae il descrittore HOG da un'immagine.

    Pipeline (come da Modulo 2 delle slide):
      1. Resize all'input standard HOG
      2. Conversione BGR→RGB (OpenCV legge in BGR!)
      3. Calcolo HOG con skimage: orientations=9, pixels_per_cell=(8,8),
    cells_per_block=(2,2), channel_axis=-1 per immagini a colori

    Args:
    img_bgr: immagine letta con cv2.imread() → formato BGR, HxWxC

    Returns:
    vettore 1D numpy con le feature HOG (float64)
    """
    
    # 1. Resize: porta l'immagine alla dimensione standard HOG
    img_resized = cv2.resize(img_bgr, HOG_IMG_SIZE)
    
    # 2. Conversione BGR → RGB
    #    IMPORTANTE: OpenCV legge in BGR, skimage/matplotlib si aspettano RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # 3. Estrazione HOG con skimage
    #    channel_axis=-1 → indica che l'ultimo asse è il canale colore
    #    L'HOG opera su tutti e 3 i canali e concatena i descrittori
    feature_vector = hog(
        img_rgb,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        channel_axis=-1,   # immagine a colori (H, W, C)
        feature_vector=True  # restituisce vettore 1D già appiattito
    )

    return feature_vector  # shape: (N,) dove N dipende dai parametri HOG

def extract_hog_dataset(dataset_path: str, classi: dict = None) -> tuple:
    
    """
    Itera su tutte le immagini del dataset ed estrae le feature HOG.

    Struttura attesa su Drive:
        dataset_path/
            Tomato_healthy/
                img001.jpg
                ...
            Tomato_Early_blight/
                img001.jpg
                ...
            Tomato_Late_blight/
                img001.jpg
                ...

    Args:
        dataset_path: percorso alla cartella radice del dataset
        classi:       dizionario {nome_cartella: label_numerica}
                      Se None, usa il dizionario CLASSI globale

    Returns:
        X: np.ndarray di shape (N, D) — matrice delle feature HOG
        y: np.ndarray di shape (N,)   — vettore delle label numeriche
    """
    if classi is None:
        classi = CLASSI
        
    X_list = []  # lista di vettori HOG
    y_list = []  # lista di label
    
    print("=" * 55)
    print("  Estrazione HOG dal dataset")
    print(f"  Dataset path: {dataset_path}")
    print(f"  Classi: {list(classi.keys())}")
    print("=" * 55)
    
    for nome_classe, label in classi.items():
        cartella = os.path.join(dataset_path, nome_classe)

        if not os.path.isdir(cartella):
            print(f"[ATTENZIONE] Cartella non trovata: {cartella}")
            continue

        # Lista di tutti i file immagine nella cartella
        file_lista = [
            f for f in os.listdir(cartella)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        print(f"\nClasse '{nome_classe}' (label={label}): {len(file_lista)} immagini")

        # Barra di progresso per monitorare l'avanzamento
        for nome_file in tqdm(file_lista, desc=f"  HOG {nome_classe[:15]}"):
            percorso = os.path.join(cartella, nome_file)

            # Carica immagine con OpenCV (legge in BGR)
            img = cv2.imread(percorso)

            if img is None:
                # Salta file corrotti o non leggibili
                print(f"  [SKIP] File non leggibile: {nome_file}")
                continue
            
            # Estrai HOG e aggiungi alle liste
            features = extract_hog_features(img)
            X_list.append(features)
            y_list.append(label)

    
    # Converti liste in array numpy
    X = np.array(X_list, dtype=np.float32)  # float32 per compatibilità sklearn
    y = np.array(y_list, dtype=np.int32)
    
    
    print(f"\n{'=' * 55}")
    print(f"  Estrazione completata!")
    print(f"  Shape X (feature matrix): {X.shape}")
    print(f"  Shape y (labels):         {y.shape}")
    print(f"  Dimensione vettore HOG:   {X.shape[1]} feature")
    print(f"  Distribuzione classi:     {np.bincount(y)}")
    print("=" * 55)

    return X, y

def visualize_hog(img_bgr: np.ndarray):
    """
    Visualizza l'immagine originale e la sua visualizzazione HOG affiancate.
    Utile per debug e per la documentazione del Technical Analysis Document.

    Args:
        img_bgr: immagine in formato BGR (output di cv2.imread)
    """
    
    from skimage.feature import hog as skimage_hog
    import matplotlib.pyplot as plt

    img_resized = cv2.resize(img_bgr, HOG_IMG_SIZE)
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # visualize=True restituisce anche l'immagine HOG renderizzata
    _, hog_image = skimage_hog(
        img_rgb,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        channel_axis=-1,
        visualize=True
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.imshow(img_rgb)
    ax1.set_title("Immagine originale (RGB)", fontsize=12)
    ax1.axis('off')
    
    ax2.imshow(hog_image, cmap='magma')
    ax2.set_title(
        f"HOG descriptor\n(orientations={HOG_ORIENTATIONS}, "
        f"cell={HOG_PIXELS_PER_CELL}, block={HOG_CELLS_PER_BLOCK})",
        fontsize=12
    )
    ax2.axis('off')
    
    plt.suptitle("Feature Extraction: Histograms of Oriented Gradients (HOG)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()