# CONTEXT.md — Plant Disease Classification
> Leggi questo file all'inizio di ogni sessione prima di fare qualsiasi cosa.

---

## Progetto
Progetto finale per l'esame **"Introduction to Computer Vision"** — Epicode Institute of Technology.
- Voto: 50% progetto pratico + 50% orale
- Deadline: 6 giorni dalla data corrente
- Livello studente: principiante assoluto di Computer Vision

---

## ✅ Stato avanzamento
> **Aggiorna questa sezione manualmente dopo ogni sessione di lavoro.**

- [ ] Struttura repo creata su GitHub
- [ ] `CONTEXT.md` nel repo
- [ ] `src/preprocessing.py` — caricamento, augmentation, dataloader
- [ ] `src/features.py` — estrazione HOG features
- [ ] `src/classical_model.py` — training e valutazione SVM
- [ ] `src/deep_model.py` — ResNet18, training loop
- [ ] `src/evaluation.py` — metriche, confusion matrix, visualizzazioni
- [ ] Notebook Colab completo e funzionante con GPU
- [ ] `requirements.txt` generato dal codice reale
- [ ] `README.md` completo con link Colab
- [ ] `docs/technical_analysis.pdf` (max 10 pagine)

---

## Obiettivo
Classificazione di malattie delle foglie di pomodoro tramite immagini, usando il dataset **PlantVillage** (Kaggle).

**Classi usate (sottoinsieme filtrato):**
- `Tomato_healthy`
- `Tomato_Early_blight`
- `Tomato_Late_blight`

*(3 classi di pomodoro — scelta deliberata per semplicità e rispetto della deadline)*

---

## Stack tecnologico
| Libreria | Uso |
|---|---|
| Python 3.x | linguaggio principale |
| OpenCV (cv2) | preprocessing, HOG, visualizzazione |
| NumPy | manipolazione array/immagini |
| Matplotlib | visualizzazione grafici e immagini |
| scikit-learn | SVM, metriche, train/test split |
| PyTorch | ResNet18 fine-tuning |
| torchvision | dataset, transforms, modelli pre-trainati |
| Google Colab | ambiente di esecuzione con GPU |

---

## Pipeline completa (obbligatoria per l'esame)

```
Dataset PlantVillage (3 classi pomodoro)
        ↓
1. PREPROCESSING
   - Resize a 224x224
   - Conversione BGR→RGB (OpenCV legge BGR!)
   - Normalizzazione con mean/std ImageNet: [0.485,0.456,0.406] / [0.229,0.224,0.225]
   - Data Augmentation (solo su train): RandomHorizontalFlip, RandomRotation(15°), ColorJitter
        ↓
2. MODELLO CLASSICO — HOG + SVM
   - Feature extraction: HOG con skimage.feature.hog() o cv2.HOGDescriptor
   - Classificatore: sklearn.svm.SVC (kernel RBF o lineare)
   - Output: Accuracy, Precision, Recall, F1, Confusion Matrix
        ↓
3. MODELLO DEEP LEARNING — ResNet18 Fine-tuned
   - Backbone: torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
   - Custom head: sostituzione dell'ultimo FC layer con Linear(512, 3)
   - Training: fine-tuning completo o solo head (da decidere in base al tempo)
   - Optimizer: Adam, lr=0.001
   - Loss: CrossEntropyLoss
   - Output: stesse metriche del modello classico
        ↓
4. POST-PROCESSING & VALUTAZIONE
   - Confronto esplicito HOG+SVM vs ResNet18
   - Visualizzazione predizioni sbagliate (failure analysis)
   - Confusion Matrix per entrambi i modelli
   - Grafici: loss curve, accuracy curve per ResNet18
```

---

## Struttura cartelle del repo

```
plant-disease-classification/
├── notebooks/
│   └── plant_disease_classification.ipynb   # Notebook Colab principale
├── src/
│   ├── preprocessing.py     # caricamento dati, augmentation, dataloader
│   ├── features.py          # estrazione HOG features
│   ├── classical_model.py   # training e valutazione SVM
│   ├── deep_model.py        # definizione ResNet18, training loop
│   └── evaluation.py        # metriche, confusion matrix, visualizzazioni
├── docs/
│   └── technical_analysis.pdf   # Technical Analysis Document (max 10 pag)
├── README.md
├── requirements.txt
└── CONTEXT.md               # questo file
```

---

## Deliverables obbligatori (checklist esame)

- [ ] Notebook Colab funzionante con GPU (link nel README)
- [ ] `requirements.txt` generato dal codice reale
- [ ] `README.md` con: titolo, overview, setup, pipeline, risultati, link Colab
- [ ] `technical_analysis.pdf` (max 10 pagine) con:
  - Problem Statement
  - Methodology (HOG+SVM e ResNet18)
  - Experimental Results (tabelle + grafici)
  - Failure Analysis ← citare bias PlantVillage
  - Ethical Considerations ← citare bias PlantVillage
- [ ] Repo GitHub pubblico con tutto il codice

---

## Note critiche per il codice

### Colab
```python
# Visualizzare immagini OpenCV in Colab (NON usare cv2.imshow!)
from google.colab.patches import cv2_imshow

# Montare Google Drive se necessario
from google.colab import drive
drive.mount('/content/drive')
```

### Download dataset da Kaggle in Colab
```python
# Metodo 1: upload manuale del file kaggle.json
from google.colab import files
files.upload()  # carica il file kaggle.json dalla macchina locale

import os
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
os.rename("kaggle.json", os.path.expanduser("~/.kaggle/kaggle.json"))
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Metodo 2: download via API (dopo aver configurato kaggle.json)
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
api.dataset_download_files(
    'abdallahalidev/plantvillage-dataset',
    path='/content/plantvillage',
    unzip=True
)

# In alternativa, via shell direttamente nel notebook:
# !pip install kaggle
# !kaggle datasets download -d abdallahalidev/plantvillage-dataset --unzip -p /content/plantvillage
```

### OpenCV BGR → RGB
```python
# OpenCV legge sempre in BGR — SEMPRE convertire prima di visualizzare
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
```

### Normalizzazione ImageNet
```python
# Valori standard per modelli pre-trainati su ImageNet
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### ResNet18 custom head — SINTASSI MODERNA (PyTorch >= 0.13)
```python
# ATTENZIONE: pretrained=True è deprecato da PyTorch 0.13 in poi
# Usare sempre la sintassi moderna con weights= per evitare warning

from torchvision.models import resnet18, ResNet18_Weights

# Carica ResNet18 pre-trainata su ImageNet (sintassi moderna)
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Sostituisce il classificatore finale con uno custom per N classi
num_features = model.fc.in_features  # 512 per ResNet18
NUM_CLASSI = 3  # modifica qui se aggiungi classi in futuro
model.fc = torch.nn.Linear(num_features, NUM_CLASSI)
```

### HOG con skimage (preferito per semplicità)
```python
from skimage.feature import hog
features = hog(img, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), channel_axis=-1)
```

---

## Dataset — Note importanti

- **Fonte:** Kaggle — `abdallahalidev/plantvillage-dataset`
- **Dimensione totale:** ~54.000 immagini, 38 classi, 14 specie
- **Subset usato:** solo 3 classi di pomodoro (filtrare all'inizio)
- **Split consigliato:** 80% train, 10% validation, 10% test
- **BIAS NOTO (da citare nella documentazione):**
  Il dataset è stato acquisito in laboratorio su sfondo uniforme. Una ricerca ha dimostrato che un modello addestrato su soli 8 pixel di background raggiunge il 49% di accuracy (vs 2.6% random). Questo significa che i modelli rischiano di imparare il background invece delle caratteristiche della foglia. Da citare esplicitamente in Failure Analysis ed Ethical Considerations.

### Espandibilità futura del dataset
La pipeline è progettata per essere facilmente espandibile. Per aggiungere classi:
1. **Modello classico (HOG+SVM):** nessuna modifica strutturale — `SVC` di scikit-learn
   gestisce nativamente la classificazione multi-classe. Basta includere le nuove cartelle
   nel caricamento e il modello si adatta automaticamente.
2. **ResNet18:** modificare una sola riga — `NUM_CLASSI` nel custom head. Il resto
   della pipeline (training loop, metriche, DataLoader) non cambia.
3. **Costo computazionale:** ogni classe aggiunta aumenta il tempo di training HOG+SVM
   in modo significativo (l'estrazione HOG è lenta su grandi volumi); ResNet18 scala
   meglio grazie alla GPU.
4. **Attenzione al class imbalance:** PlantVillage non ha lo stesso numero di immagini
   per ogni classe — con più classi il problema si amplifica. Considerare
   `class_weight='balanced'` in SVC e WeightedRandomSampler in PyTorch.

---

## Metriche obbligatorie

```python
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix,
                              classification_report)

# Per entrambi i modelli (HOG+SVM e ResNet18)
print(classification_report(y_true, y_pred,
      target_names=['healthy','early_blight','late_blight']))
```

---

## Argomenti del corso (per coerenza con le lezioni)

Il professore conosce queste slide. Quando scrivi codice o commenti, usa la terminologia del corso:

| Modulo | Argomenti chiave rilevanti per il progetto |
|---|---|
| 1 — Image Processing Fundamentals | Pipeline imaging, NumPy array, OpenCV, BGR→RGB, imread/imwrite |
| 2 — Segmentation & Feature Extraction | HOG, SIFT, SURF, ORB, Harris corners, thresholding, Canny |
| 3 — Classification & Recognition | Supervised learning, HOG+SVM pipeline, data augmentation, overfitting, k-NN |
| 4 — Object Detection & Video | IoU, NMS, R-CNN family, YOLO, SSD (contesto teorico orale) |
| 5 — 3D Vision & Image Generation | GANs, Diffusion Models, VAEs (contesto teorico orale) |
| 6 — Advanced Topics | U-Net, Mask R-CNN, ViT, CLIP, Latent Diffusion, scene understanding (orale) |

**Collegamento lezioni nel codice:** quando pertinente aggiungi commenti tipo:
```python
# Come visto nel Modulo 3: estrazione HOG features per il classificatore SVM
# Come visto nel Modulo 1: conversione BGR→RGB necessaria per Matplotlib
```

---

## Regole di sviluppo

1. **Lingua:** tutto in italiano — commenti, docstring, output print, documentazione
2. **Modularità:** ogni file `src/` ha una responsabilità sola — niente monoliti
3. **Semplicità prima di tutto:** soluzione semplice e funzionante > soluzione elegante e rotta
4. **Deadline:** 6 giorni — se c'è una scelta tra veloce e perfetto, scegli veloce
5. **Commenti:** ogni funzione ha docstring + commenti inline sulle righe non ovvie
6. **Git:** commit frequenti con messaggi descrittivi in italiano
7. **Colab-first:** tutto il codice deve girare su Colab senza modifiche locali

---

## Flusso di lavoro consigliato

```
Claude Code (locale)          Colab (GPU)               Progetto Claude (strategia)
─────────────────────         ──────────────            ──────────────────────────
Crea struttura repo      →    Incolla e testa      →    Chiedi spiegazioni/codice
Gestisce file .py             il notebook               Scrivi documentazione
Fa commit/push git            con GPU reale             Prepara README e PDF
Genera requirements.txt       Salva risultati           Simula domande orale
```

---
> **Istruzioni per Claude Code:** all'inizio di ogni sessione leggi questo file,
> controlla lo stato avanzamento e chiedi all'utente quale task vuole completare oggi.