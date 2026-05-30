# src/deep_model.py
"""
Modulo 3 - Deep Learning: ResNet18 Fine-Tuning con PyTorch
Come visto a lezione, applichiamo la strategia di Transfer Learning in due fasi:
  Fase 1 (Feature Extraction): congeliamo tutti i layer di ResNet18, alleniamo
          solo il nuovo classification head (fc layer custom).
  Fase 2 (Fine-Tuning):        scongeliamo tutto il modello e continuiamo
          il training con un learning rate molto più basso per evitare
          il "catastrophic forgetting" descritto nel Modulo 3.
"""

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.models import ResNet18_Weights

# ─────────────────────────────────────────────────────────────
# 1. BUILD_RESNET18
# ─────────────────────────────────────────────────────────────

def build_resnet18(num_classi: int = 3) -> nn.Module:
    """
    Carica ResNet18 pre-trainato su ImageNet e sostituisce il layer
    finale (fc) con un classificatore custom per le nostre 3 classi.

    Architettura ResNet18 (come da Modulo 3):
      - Layer 1-4: blocchi residui convoluzionali (pesi ImageNet)
      - AdaptiveAvgPool2d: riduce ogni feature map a 1×1
      - fc (sostituito): Linear(512, num_classi) → nostro classificatore

    Args:
        num_classi: numero di classi del problema (default 3)

    Returns:
        model: ResNet18 con fc custom, pronto per il fine-tuning
    """
    
    # Sintassi moderna (PyTorch >= 0.13): weights= invece di pretrained=True
    # ResNet18_Weights.IMAGENET1K_V1 carica i pesi allenati su ImageNet
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    # Il layer fc originale è Linear(512, 1000) — 1000 classi ImageNet
    # Lo sostituiamo con Linear(512, num_classi) per il nostro task
    # in_features=512 è fisso nell'architettura ResNet18
    num_features = model.fc.in_features          # = 512
    model.fc = nn.Linear(num_features, num_classi)
    
    return model

# ─────────────────────────────────────────────────────────────
# 2. TRAIN_EPOCH
# ─────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    
    """
    Esegue UNA singola epoca di training.

    Per ogni batch del DataLoader:
      1. Sposta i dati sulla GPU (device)
      2. Forward pass: calcola le predizioni del modello
      3. Calcola la loss (CrossEntropyLoss)
      4. Backward pass: calcola i gradienti (backpropagation)
      5. Aggiorna i pesi con l'optimizer (Adam/SGD)
      6. Azzera i gradienti per il batch successivo

    Args:
        model:     il modello ResNet18
        loader:    DataLoader del training set
        optimizer: ottimizzatore (Adam)
        criterion: funzione di loss (CrossEntropyLoss)
        device:    'cuda' se GPU disponibile, altrimenti 'cpu'

    Returns:
        (loss_media, accuracy_media) dell'epoca
    """
    
    model.train()   # modalità training: attiva dropout, batch norm in training mode

    loss_totale   = 0.0
    corretti      = 0
    totale        = 0
    
    for immagini, etichette in loader:
        # Sposta batch su GPU
        immagini  = immagini.to(device)
        etichette = etichette.to(device)

        # Azzera i gradienti accumulati dal batch precedente
        optimizer.zero_grad()

        # Forward pass: ottieni i logits (score per ogni classe)
        output = model(immagini)
        
        # Calcola la loss tra predizioni e etichette reali
        loss = criterion(output, etichette)

        # Backward pass: calcola i gradienti rispetto alla loss
        loss.backward()

        # Aggiorna i pesi del modello
        optimizer.step()

        # Accumula statistiche
        loss_totale += loss.item() * immagini.size(0)
        _, predetti  = torch.max(output, 1)       # classe con score più alto
        corretti    += (predetti == etichette).sum().item()
        totale      += etichette.size(0)
        
    loss_media     = loss_totale / totale
    accuracy_media = corretti / totale

    return loss_media, accuracy_media

# ─────────────────────────────────────────────────────────────
# 3. VAL_EPOCH
# ─────────────────────────────────────────────────────────────

def val_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    
    """
    Esegue UNA singola epoca di validazione (senza aggiornare i pesi).

    Identica a train_epoch ma:
      - model.eval(): disattiva dropout e batch norm in training mode
      - torch.no_grad(): non calcola i gradienti → risparmio di memoria GPU

    Serve a monitorare se il modello sta generalizzando o andando
    in overfitting (come spiegato nel Modulo 3: training loss scende
    ma validation loss sale → segnale di overfitting).

    Args:
        model:     il modello ResNet18
        loader:    DataLoader del validation set
        criterion: funzione di loss (CrossEntropyLoss)
        device:    'cuda' o 'cpu'

    Returns:
        (loss_media, accuracy_media) sulla validation
    """
    
    model.eval()   # modalità eval: disattiva dropout e batch norm training

    loss_totale = 0.0
    corretti    = 0
    totale      = 0
    
    with torch.no_grad():   # nessun calcolo di gradienti → più veloce e leggero
        for immagini, etichette in loader:
            immagini  = immagini.to(device)
            etichette = etichette.to(device)

            output = model(immagini)
            loss   = criterion(output, etichette)

            loss_totale += loss.item() * immagini.size(0)
            _, predetti  = torch.max(output, 1)
            corretti    += (predetti == etichette).sum().item()
            totale      += etichette.size(0)
            
    loss_media     = loss_totale / totale
    accuracy_media = corretti / totale

    return loss_media, accuracy_media


# ─────────────────────────────────────────────────────────────
# 4. TRAINING_LOOP
# ─────────────────────────────────────────────────────────────

def training_loop(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    save_path: str,
    epoche_fase1: int = 10,
    epoche_fase2: int = 10,
    lr_fase1: float = 1e-3,
    lr_fase2: float = 1e-4,
    device: torch.device = None
) -> dict:
    
    """
    Esegue il training completo in due fasi (come da Modulo 3):

    ── FASE 1 (Feature Extraction) ──────────────────────────────
    Congela tutti i layer di ResNet18 tranne fc.
    Solo il classificatore custom viene allenato.
    lr = 0.001 (standard per un layer appena inizializzato)
    Obiettivo: portare fc a uno stato ragionevole prima di scongelare
               tutto — evita che i gradienti casuali di fc distruggano
               i pesi pre-trainati (catastrophic forgetting).
    
     ── FASE 2 (Fine-Tuning) ─────────────────────────────────────
    Scongela TUTTI i layer del modello.
    lr = 0.0001 (10× più basso della fase 1)
    Obiettivo: adattare delicatamente i layer profondi al nostro dataset
               senza distruggere le feature già apprese su ImageNet.

    In entrambe le fasi: salva il modello ogni volta che la
    validation accuracy migliora (best model checkpoint).

    Args:
        model:        ResNet18 costruito con build_resnet18()
        train_loader: DataLoader del training set
        val_loader:   DataLoader del validation set
        save_path:    percorso dove salvare il best model (.pth)
        epoche_fase1: numero di epoche per la Fase 1 (default 10)
        epoche_fase2: numero di epoche per la Fase 2 (default 10)
        lr_fase1:     learning rate Fase 1 (default 0.001)
        lr_fase2:     learning rate Fase 2 (default 0.0001)
        device:       'cuda' o 'cpu' (auto-detect se None)

    Returns:
        history: dizionario con liste di loss e accuracy per ogni epoca
                 {'train_loss', 'val_loss', 'train_acc', 'val_acc'}
    """
    
    # Auto-detect GPU
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device in uso: {device}")

    model = model.to(device)
    
    # Funzione di loss: CrossEntropyLoss — standard per classificazione multi-classe
    criterion = nn.CrossEntropyLoss()

    # Dizionario per tracciare l'andamento del training (servirà per i grafici)
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc':  [], 'val_acc':  []
    }
    
    # Migliore accuracy di validazione vista finora (per il salvataggio del best model)
    migliore_val_acc  = 0.0
    migliori_pesi     = copy.deepcopy(model.state_dict())
    
    # ── FASE 1: Feature Extraction ────────────────────────────
    print("\n" + "="*55)
    print("  FASE 1 — Feature Extraction")
    print(f"  Epoche: {epoche_fase1}  |  lr: {lr_fase1}")
    print("="*55)
    
    # Congela TUTTI i parametri del modello
    for parametro in model.parameters():
        parametro.requires_grad = False
        
    # Scongela SOLO il layer fc (il nostro classificatore custom)
    # Questi sono gli unici pesi che verranno aggiornati nella Fase 1
    for parametro in model.fc.parameters():
        parametro.requires_grad = True
        
    # Optimizer: passa solo i parametri che hanno requires_grad=True
    # → Adam solo sui pesi di fc
    optimizer_fase1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_fase1
    )
    
    for epoca in range(1, epoche_fase1 + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_fase1, criterion, device)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Fase1 Ep {epoca:2d}/{epoche_fase1} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Salva il modello se questa è la migliore val accuracy finora
        if val_acc > migliore_val_acc:
            migliore_val_acc = val_acc
            migliori_pesi    = copy.deepcopy(model.state_dict())
            torch.save(migliori_pesi, save_path)
            print(f"  ✅ Best model salvato (val_acc: {migliore_val_acc:.4f})")
            
    # ── FASE 2: Fine-Tuning ───────────────────────────────────
    print("\n" + "="*55)
    print("  FASE 2 — Fine-Tuning (tutti i layer)")
    print(f"  Epoche: {epoche_fase2}  |  lr: {lr_fase2}")
    print("="*55)

    # Scongela TUTTI i parametri del modello
    for parametro in model.parameters():
        parametro.requires_grad = True
    
    # Nuovo optimizer con lr molto più basso (come da slide Modulo 3:
    # "use a very low learning rate to avoid catastrophic forgetting")
    optimizer_fase2 = optim.Adam(model.parameters(), lr=lr_fase2)

    for epoca in range(1, epoche_fase2 + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer_fase2, criterion, device)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f"Fase2 Ep {epoca:2d}/{epoche_fase2} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        if val_acc > migliore_val_acc:
            migliore_val_acc = val_acc
            migliori_pesi    = copy.deepcopy(model.state_dict())
            torch.save(migliori_pesi, save_path)
            print(f"  ✅ Best model salvato (val_acc: {migliore_val_acc:.4f})")

    # Carica i migliori pesi nel modello prima di restituirlo
    model.load_state_dict(migliori_pesi)
    print(f"\nTraining completato. Miglior val accuracy: {migliore_val_acc:.4f}")

    return history

# ─────────────────────────────────────────────────────────────
# 5. PLOT_TRAINING_HISTORY
# ─────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None):
    
    """
    Genera due grafici affiancati: curva della Loss e curva dell'Accuracy.
    Entrambi mostrano train vs validation — utile per rilevare overfitting.

    Come spiegato nel Modulo 3:
      - Se train_acc cresce ma val_acc si appiattisce → overfitting
      - Le due curve dovrebbero convergere per un modello che generalizza bene

    Il punto di demarcazione tra Fase 1 e Fase 2 viene segnato con una
    linea tratteggiata verticale per rendere visibile l'effetto del fine-tuning.

    Args:
        history:   dizionario restituito da training_loop()
        save_path: percorso opzionale dove salvare il grafico (.png)
    """
    
    epoche_totali = len(history['train_loss'])
    asse_x        = range(1, epoche_totali + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ResNet18 Fine-Tuning — Andamento Training', fontsize=14, fontweight='bold')

    # ── Grafico Loss ──────────────────────────────────────────
    ax1.plot(asse_x, history['train_loss'], label='Train Loss', color='royalblue',  linewidth=2)
    ax1.plot(asse_x, history['val_loss'],   label='Val Loss',   color='tomato',     linewidth=2)
    ax1.axvline(x=10.5, color='gray', linestyle='--', linewidth=1, label='Fine-tuning inizia')
    ax1.set_title('Loss per Epoca')
    ax1.set_xlabel('Epoca')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ── Grafico Accuracy ──────────────────────────────────────
    ax2.plot(asse_x, [a * 100 for a in history['train_acc']], label='Train Accuracy', color='royalblue', linewidth=2)
    ax2.plot(asse_x, [a * 100 for a in history['val_acc']],   label='Val Accuracy',   color='tomato',    linewidth=2)
    ax2.axvline(x=10.5, color='gray', linestyle='--', linewidth=1, label='Fine-tuning inizia')
    ax2.set_title('Accuracy per Epoca')
    ax2.set_xlabel('Epoca')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grafico salvato in: {save_path}")

    plt.show()