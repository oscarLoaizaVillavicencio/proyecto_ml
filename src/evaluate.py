"""Métricas para evaluar modelos de clasificación."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def evaluar(modelo, X, y, umbral: float = 0.5) -> dict:
    """Calcula las métricas principales. El umbral decide cuándo predecir churn."""
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba >= umbral).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def comparar(modelos: dict, X, y) -> pd.DataFrame:
    """Tabla comparativa de varios modelos ya entrenados."""
    filas = {nombre: evaluar(m, X, y) for nombre, m in modelos.items()}
    return pd.DataFrame(filas).T.round(3)


def matriz_confusion(modelo, X, y, umbral: float = 0.5) -> pd.DataFrame:
    """Matriz de confusión con etiquetas legibles."""
    pred = (modelo.predict_proba(X)[:, 1] >= umbral).astype(int)
    return pd.DataFrame(
        confusion_matrix(y, pred),
        index=["real: se queda", "real: se va"],
        columns=["pred: se queda", "pred: se va"],
    )