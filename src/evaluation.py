"""
evaluation.py
=============
Giorno 5 — Valutazione e confronto tra modello classico (HOG+SVM)
e modello deep learning (ResNet18 fine-tuned).
 
Modulo corso di riferimento: Modulo 3 (valutazione, confusion matrix,
F1-score, error analysis, post-processing).
 
Funzioni:
    - get_predictions_svm       : predizioni SVM sul test set
    - get_predictions_resnet    : predizioni ResNet18 sul test set
    - plot_confusion_matrix     : heatmap normalizzata (seaborn)
    - compare_models            : tabella comparativa affiancata
    - show_errors               : visualizzazione immagini classificate male
"""

# ── Librerie standard ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import torch
 
# Nota Colab: cv2_imshow non serve qui perché usiamo matplotlib per tutti i plot
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 1. PREDIZIONI SVM
# ══════════════════════════════════════════════════════════════════════════════
 
def get_predictions_svm(model, X_test):
    """
    Restituisce le predizioni del modello SVM sul test set.
 
    Come visto nel Modulo 3, dopo il training basta chiamare .predict()
    sul modello scikit-learn — la pipeline HOG+SVM è già addestrata.
 
    Args:
        model  : modello SVM addestrato (sklearn.svm.SVC caricato con joblib)
        X_test : matrice numpy di feature HOG estratte dal test set
                 shape attesa: (n_campioni, n_feature_hog)
 
    Returns:
        y_pred : array numpy con le label predette (interi 0, 1, 2)
    """
    print("[SVM] Generazione predizioni sul test set...")
    y_pred = model.predict(X_test)
    print(f"[SVM] Predizioni generate: {len(y_pred)} campioni")
    return y_pred
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 2. PREDIZIONI RESNET18
# ══════════════════════════════════════════════════════════════════════════════
 
def get_predictions_resnet(model, test_loader, device):
    """
    Restituisce le predizioni di ResNet18 sul test set.
 
    Come spiegato nel Modulo 3 (Transfer Learning):
    - model.eval() disattiva Dropout e BatchNorm (comportamento inferenza)
    - torch.no_grad() evita il calcolo del grafo computazionale (risparmio
      memoria GPU significativo durante la fase di valutazione)
    - raccogliamo predizioni batch per batch e le concateniamo alla fine
 
    Args:
        model       : ResNet18 fine-tuned caricato da Drive (torch.nn.Module)
        test_loader : DataLoader del test set (da preprocessing.py)
        device      : torch.device — 'cuda' se GPU disponibile, 'cpu' altrimenti
 
    Returns:
        y_true : array numpy con le label reali
        y_pred : array numpy con le label predette
    """
    
    print(f"[ResNet18] Generazione predizioni su device: {device}")
    model.eval()          # modalità inferenza: disattiva dropout / batchnorm
    model.to(device)
 
    lista_pred  = []      # accumulatori batch
    lista_label = []
 
    with torch.no_grad():  # nessun calcolo del gradiente → risparmio memoria
        for immagini, label in test_loader:
            immagini = immagini.to(device)
            label    = label.to(device)
 
            output = model(immagini)               # forward pass
            _, predetti = torch.max(output, dim=1) # classe con score massimo
 
            lista_pred.append(predetti.cpu().numpy())
            lista_label.append(label.cpu().numpy())
            
 
    y_pred = np.concatenate(lista_pred)
    y_true = np.concatenate(lista_label)
 
    print(f"[ResNet18] Predizioni generate: {len(y_pred)} campioni")
    return y_true, y_pred
 
 
# ══════════════════════════════════════════════════════════════════════════════
# 3. CONFUSION MATRIX NORMALIZZATA
# ══════════════════════════════════════════════════════════════════════════════
 
def plot_confusion_matrix(y_true, y_pred, classi, titolo, salva_path=None):
    """
    Visualizza la Confusion Matrix normalizzata come heatmap seaborn.
 
    Come visto nel Modulo 3 (Task 1 del laboratorio):
    - La normalizzazione (valori 0-1) mostra le proporzioni degli errori
      invece dei conteggi assoluti → più leggibile con classi bilanciate
    - Le celle sulla diagonale = predizioni corrette
    - Le celle fuori diagonale = errori (utili per la Failure Analysis)
 
    Args:
        y_true     : array numpy con label reali
        y_pred     : array numpy con label predette
        classi     : lista di stringhe con i nomi delle classi
                     es. ['Tomato_Healthy', 'Early_Blight', 'Late_Blight']
        titolo     : stringa titolo del grafico
        salva_path : path (stringa) dove salvare il grafico come PNG
                     (None = non salvare, solo mostrare)
    """
    # Calcolo matrice normalizzata per riga (ogni riga somma a 1)
    cm = confusion_matrix(y_true, y_pred, normalize='true')
 
    fig, ax = plt.subplots(figsize=(8, 6))
 
    sns.heatmap(
        cm,
        annot=True,           # mostra i valori nelle celle
        fmt='.2f',            # formato decimale a 2 cifre
        cmap='Blues',         # palette blu: più scuro = più predizioni
        xticklabels=classi,
        yticklabels=classi,
        linewidths=0.5,
        ax=ax
    )
    
     
    ax.set_title(titolo, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Label Reale', fontsize=12)
    ax.set_xlabel('Label Predetta', fontsize=12)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
 
    plt.tight_layout()
 
    # Salvataggio per il documento PDF del Giorno 6
    if salva_path:
        plt.savefig(salva_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Confusion matrix salvata in: {salva_path}")
 
    plt.show()
 
    # Stampa anche il classification report testuale per completezza
    print(f"\n{'='*60}")
    print(f"Classification Report — {titolo}")
    print('='*60)
    print(classification_report(y_true, y_pred, target_names=classi))
     
 
# ══════════════════════════════════════════════════════════════════════════════
# 4. CONFRONTO MODELLI
# ══════════════════════════════════════════════════════════════════════════════
 
def _calcola_metriche(y_true, y_pred, nome):
    """
    Funzione di supporto: calcola le metriche aggregate per un modello.
    Usa average='weighted' per tener conto del bilanciamento delle classi.
 
    Args:
        y_true : array label reali
        y_pred : array label predette
        nome   : nome del modello (stringa, per il dizionario)
 
    Returns:
        dict con accuracy, precision, recall, f1
    """
    return {
        'Modello'  : nome,
        'Accuracy' : round(accuracy_score(y_true, y_pred) * 100, 2),
        'Precision': round(precision_score(y_true, y_pred, average='weighted') * 100, 2),
        'Recall'   : round(recall_score(y_true, y_pred, average='weighted') * 100, 2),
        'F1-Score' : round(f1_score(y_true, y_pred, average='weighted') * 100, 2),
    }
    
    
def compare_models(y_true_svm, y_pred_svm, y_true_resnet, y_pred_resnet,
                   salva_path=None):
    """
    Calcola e stampa una tabella comparativa affiancata tra SVM e ResNet18.
 
    Il confronto esplicito classico vs deep learning è un requisito
    obbligatorio dell'esame (Final Exam requirements).
 
    Args:
        y_true_svm    : label reali test set SVM
        y_pred_svm    : label predette SVM
        y_true_resnet : label reali test set ResNet
        y_pred_resnet : label predette ResNet
        salva_path    : path PNG per salvare il grafico comparativo (opzionale)
 
    Returns:
        dict con le metriche di entrambi i modelli
    """
    
    metriche_svm    = _calcola_metriche(y_true_svm,    y_pred_svm,    'HOG + SVM')
    metriche_resnet = _calcola_metriche(y_true_resnet, y_pred_resnet, 'ResNet18')
 
    # ── Stampa tabella testuale ──────────────────────────────────────────────
    separatore = '-' * 52
    print(f"\n{'='*52}")
    print(f"{'CONFRONTO MODELLI':^52}")
    print(f"{'='*52}")
    print(f"{'Metrica':<15} {'HOG + SVM':>15} {'ResNet18':>15}")
    print(separatore)
 
    chiavi = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for k in chiavi:
        print(f"{k:<15} {metriche_svm[k]:>14.2f}% {metriche_resnet[k]:>14.2f}%")
 
    print(separatore)
    
     # Evidenzia il modello migliore per F1
    if metriche_resnet['F1-Score'] > metriche_svm['F1-Score']:
        delta = metriche_resnet['F1-Score'] - metriche_svm['F1-Score']
        print(f"\n→ ResNet18 supera HOG+SVM di {delta:.2f}% su F1-Score")
    else:
        delta = metriche_svm['F1-Score'] - metriche_resnet['F1-Score']
        print(f"\n→ HOG+SVM supera ResNet18 di {delta:.2f}% su F1-Score")
 
    # ── Grafico a barre comparativo ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    
    x      = np.arange(len(chiavi))
    width  = 0.35
    valori_svm    = [metriche_svm[k]    for k in chiavi]
    valori_resnet = [metriche_resnet[k] for k in chiavi]
 
    barre_svm    = ax.bar(x - width/2, valori_svm,    width, label='HOG + SVM',
                          color='steelblue', alpha=0.85)
    barre_resnet = ax.bar(x + width/2, valori_resnet, width, label='ResNet18',
                          color='coral', alpha=0.85)
    
    # Etichette valori sopra le barre
    for barra in barre_svm:
        ax.text(barra.get_x() + barra.get_width()/2,
                barra.get_height() + 0.5,
                f"{barra.get_height():.1f}%",
                ha='center', va='bottom', fontsize=9)
 
    for barra in barre_resnet:
        ax.text(barra.get_x() + barra.get_width()/2,
                barra.get_height() + 0.5,
                f"{barra.get_height():.1f}%",
                ha='center', va='bottom', fontsize=9)
        
    ax.set_title('Confronto Metriche: HOG+SVM vs ResNet18',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Valore (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(chiavi, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
 
    plt.tight_layout()
 
    if salva_path:
        plt.savefig(salva_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Grafico comparativo salvato in: {salva_path}")
 
    plt.show()
 
    return {'svm': metriche_svm, 'resnet': metriche_resnet}


def show_errors(model, test_loader, classi, device, n=6, salva_path=None):
    """
    Visualizza le prime n immagini classificate male da ResNet18.
 
    Come visto nel Modulo 3 (Error Analysis):
    "Le metriche quantitative (F1, Accuracy) dicono QUANTO il modello
    sbaglia, ma guardare gli errori dice PERCHÉ sbaglia."
 
    Questo passaggio è fondamentale per la Failure Analysis del documento:
    - Confusione tra Early Blight e Late Blight → lesioni fogliari simili
    - Confusione con lo sfondo laboratorio → bias PlantVillage
    → Rischio scarsa generalizzazione su immagini "in campo aperto"
 
    Args:
        model       : ResNet18 fine-tuned (torch.nn.Module)
        test_loader : DataLoader del test set
        classi      : lista nomi delle classi
        device      : torch.device
        n           : numero massimo di errori da visualizzare (default: 6)
        salva_path  : path PNG per salvare la figura (opzionale)
 
    Note:
        - Le immagini vengono de-normalizzate prima di mostrarle
          (invertendo la normalizzazione ImageNet applicata in preprocessing.py)
        - OpenCV non serve qui: usiamo matplotlib direttamente su tensori
    """
    
    print(f"[Error Analysis] Ricerca delle prime {n} predizioni errate...")
    model.eval()
    model.to(device)
 
    # Parametri di de-normalizzazione ImageNet (inverso del preprocessing)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
 
    immagini_errate = []   # (tensore_img, label_vera, label_predetta)
    
    with torch.no_grad():
        for immagini, label in test_loader:
            immagini_gpu = immagini.to(device)
            output       = model(immagini_gpu)
            _, predetti  = torch.max(output, dim=1)
 
            predetti_cpu = predetti.cpu()
            label_cpu    = label.cpu()
 
            # Seleziona solo le predizioni ERRATE
            mask_errori = (predetti_cpu != label_cpu)
            idx_errori  = mask_errori.nonzero(as_tuple=False).squeeze(1)
 
            for i in idx_errori.tolist():
                if len(immagini_errate) >= n:
                    break
                immagini_errate.append((
                    immagini[i],          # tensore normalizzato
                    label_cpu[i].item(),
                    predetti_cpu[i].item()
                ))
 
            if len(immagini_errate) >= n:
                break
 
    n_trovati = len(immagini_errate)
    if n_trovati == 0:
        print("[Error Analysis] Nessun errore trovato sul test set. Ottimo!")
        return
 
    print(f"[Error Analysis] Trovati {n_trovati} errori da visualizzare.")
    
    
    # ── Plot degli errori ────────────────────────────────────────────────────
    n_cols = min(3, n_trovati)
    n_rows = (n_trovati + n_cols - 1) // n_cols  # ceiling division
 
    fig, assi = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4 * n_rows))
    assi = np.array(assi).flatten()  # uniforma a 1D anche se n_rows=1
 
    for idx, (tensore, vera, predetta) in enumerate(immagini_errate):
        # De-normalizzazione: recupera l'immagine originale leggibile
        img = tensore.cpu().numpy().transpose(1, 2, 0)  # (C,H,W) → (H,W,C)
        img = (img * std + mean).clip(0, 1)             # de-normalize
 
        assi[idx].imshow(img)
        assi[idx].set_title(
            f"Vera: {classi[vera]}\nPredetta: {classi[predetta]}",
            fontsize=9,
            color='red',
            fontweight='bold'
        )
        assi[idx].axis('off')
 
    # Nascondi assi vuoti se n_trovati < n_rows * n_cols
    for idx in range(n_trovati, len(assi)):
        assi[idx].axis('off')
 
    plt.suptitle('Error Analysis — Immagini classificate male (ResNet18)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
 
    if salva_path:
        plt.savefig(salva_path, dpi=150, bbox_inches='tight')
        print(f"[Plot] Error analysis salvata in: {salva_path}")
 
    plt.show()