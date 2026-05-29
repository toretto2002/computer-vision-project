# src/classical_model.py
"""
Modulo 3 - Modello Classico: SVM (Support Vector Machine)
Come visto a lezione, l'SVM trova l'iperpiano ottimale che massimizza
il margine tra le classi. Il kernel RBF gestisce la non-linearità
dello spazio delle feature HOG.
"""

import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm          import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline      import Pipeline
from sklearn.metrics       import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ─────────────────────────────────────────────
# NOMI DELLE CLASSI (per i report e i grafici)
# ─────────────────────────────────────────────
NOMI_CLASSI = ["Healthy", "Early Blight", "Late Blight"]

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save_path: str = "/content/drive/MyDrive/plant_disease_cv_project/models/svm_hog.pkl",
    C: float = 1.0,
    kernel: str = "rbf"
) -> Pipeline:
    
    """
    Addestra un classificatore SVM sulle feature HOG.

    Pipeline sklearn (Modulo 3):
        1. StandardScaler: normalizza le feature HOG (media=0, std=1)
           → fondamentale per SVM: le feature devono essere sulla stessa scala
        2. SVC (Support Vector Classifier):
           - kernel='rbf': Radial Basis Function, gestisce separazioni non lineari
           - C=1.0: parametro di regolarizzazione (trade-off margine/errori)
           - class_weight='balanced': compensa eventuali squilibri di classe

    Args:
        X_train:   feature HOG, shape (N_train, D)
        y_train:   label numeriche, shape (N_train,)
        save_path: percorso dove salvare il modello con joblib
        C:         parametro di regolarizzazione SVM
        kernel:    tipo di kernel ('rbf', 'linear', 'poly')

    Returns:
        pipeline: oggetto sklearn Pipeline già addestrato
    """
    
    print("=" * 55)
    print("  Training SVM (HOG + StandardScaler + SVC)")
    print(f"  Kernel: {kernel}  |  C: {C}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Feature dimensionality: {X_train.shape[1]}")
    print("=" * 55)
    
    # Pipeline: StandardScaler → SVC
    # Lo scaler è incluso nella pipeline così da evitare data leakage:
    # il fit dello scaler avviene solo sui dati di training
    pipeline = Pipeline([
        ("scaler", StandardScaler()),          # normalizzazione feature
        ("svm",    SVC(
            kernel=kernel,
            C=C,
            class_weight="balanced",           # robustezza a classi sbilanciate
            random_state=42,
            probability=False                  # non servono probabilità per SVM base
        ))
    ])
    
    print("\nAddestramento in corso... (può richiedere qualche minuto)")
    pipeline.fit(X_train, y_train)
    print("Addestramento completato!")

    # Salva il modello su Drive con joblib
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    print(f"\nModello salvato in: {save_path}")

    return pipeline

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    nome_modello: str = "SVM + HOG"
) -> tuple:
    
    """
    Valuta il modello e stampa le metriche complete.

    Metriche obbligatorie (Modulo 3 + requisiti esame):
        - Accuracy globale
        - Precision, Recall, F1-score per classe (classification_report)
        - Confusion Matrix

    Args:
        model:        modello sklearn già addestrato (Pipeline)
        X_test:       feature HOG del test set, shape (N_test, D)
        y_test:       label vere, shape (N_test,)
        nome_modello: stringa per l'intestazione del report

    Returns:
        y_pred: array delle predizioni
        cm:     confusion matrix (np.ndarray)
    """
    
    # Predizioni sul test set
    y_pred = model.predict(X_test)

    # ── Accuracy ─────────────────────────────
    acc = accuracy_score(y_test, y_pred)

    # ── Classification Report ────────────────
    report = classification_report(
        y_test, y_pred,
        target_names=NOMI_CLASSI,
        digits=4  # 4 decimali per maggiore precisione
    )
    
    # ── Confusion Matrix ─────────────────────
    cm = confusion_matrix(y_test, y_pred)

    # Stampa tutto in modo leggibile
    print("\n" + "=" * 55)
    print(f"  Risultati: {nome_modello}")
    print("=" * 55)
    print(f"\n  Accuracy:  {acc:.4f}  ({acc*100:.2f}%)")
    print(f"\n{report}")

    return y_pred, cm

def plot_confusion_matrix(
    cm: np.ndarray,
    classi: list = None,
    titolo: str = "Confusion Matrix — SVM + HOG",
    save_path: str = None
):
    
    """
    Visualizza la confusion matrix come heatmap seaborn normalizzata.

    Args:
        cm:        confusion matrix (output di sklearn confusion_matrix)
        classi:    lista dei nomi delle classi
        titolo:    titolo del grafico
        save_path: se specificato, salva il grafico come PNG
    """
    
    if classi is None:
        classi = NOMI_CLASSI

    # Normalizza: ogni cella mostra la percentuale rispetto alla riga
    # (quante predizioni corrette su tutte le immagini di quella classe)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Confusion Matrix assoluta ─────────────
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classi, yticklabels=classi,
        ax=axes[0], linewidths=0.5
    )
    
    axes[0].set_title(f"{titolo}\n(valori assoluti)", fontsize=12)
    axes[0].set_xlabel("Predizione",  fontsize=11)
    axes[0].set_ylabel("Label vera",  fontsize=11)
    
    sns.heatmap(
        cm_norm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=classi, yticklabels=classi,
        ax=axes[1], linewidths=0.5, vmin=0, vmax=1
    )
    axes[1].set_title(f"{titolo}\n(normalizzata per classe)", fontsize=12)
    axes[1].set_xlabel("Predizione", fontsize=11)
    axes[1].set_ylabel("Label vera", fontsize=11)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Grafico salvato in: {save_path}")

    plt.show()
    
    
def load_svm(
    load_path: str = "/content/drive/MyDrive/plant_disease_cv_project/models/svm_hog.pkl"
) -> Pipeline:
    """
    Carica un modello SVM precedentemente salvato con joblib.

    Args:
        load_path: percorso del file .pkl

    Returns:
        pipeline sklearn caricata
    """
    pipeline = joblib.load(load_path)
    print(f"Modello SVM caricato da: {load_path}")
    return pipeline