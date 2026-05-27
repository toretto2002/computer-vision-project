# =============================================================
# config.py — Configurazione centralizzata del progetto
# Tutti i path e gli iperparametri sono definiti qui.
# Il resto del codice importa da questo file, non definisce
# path propri.
# =============================================================

import os

# -------------------------------------------------------------
# RILEVAMENTO AMBIENTE (Colab vs Locale)
# -------------------------------------------------------------
# os.path.exists controlla se la cartella /content/drive esiste.
# Esiste solo su Google Colab quando Drive è montato.

if os.path.exists("/content/drive"):
    BASE_PATH = "/content/drive/MyDrive/plant_disease_cv_project/"
else:
    # Fallback locale — non usato in pratica ma rende il codice portabile
    BASE_PATH = "./data/"

# -------------------------------------------------------------
# PATH PRINCIPALI
# -------------------------------------------------------------
DATASET_PATH   = os.path.join(BASE_PATH, "dataset/")
MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models/")

# -------------------------------------------------------------
# CLASSI DEL DATASET
# Ordine importante: corrisponde alle label numeriche 0, 1, 2
# -------------------------------------------------------------
CLASSI = [
    "Tomato_healthy",       # label 0
    "Tomato_Early_blight",  # label 1
    "Tomato_Late_blight"    # label 2
]

# -------------------------------------------------------------
# IPERPARAMETRI
# -------------------------------------------------------------
BATCH_SIZE   = 32
IMAGE_SIZE   = 224      # ResNet18 vuole 224x224
EPOCHS       = 20       # 10 frozen + 10 unfrozen (Giorno 4)
LR_FASE1     = 0.001    # learning rate con layer congelati
LR_FASE2     = 0.0001   # learning rate fine-tuning completo
NUM_CLASSI   = len(CLASSI)  # 3

# -------------------------------------------------------------
# NORMALIZZAZIONE IMAGENET
# Valori standard per modelli pre-addestrati su ImageNet.
# ResNet18 si aspetta esattamente questi valori.
# -------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]