"""Métricas y análisis de umbral para modelos de clasificación."""
import numpy as np
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


def tabla_umbrales(y, proba, umbrales=None) -> pd.DataFrame:
    """Precision, recall y clientes marcados para distintos umbrales."""
    y = np.asarray(y)
    if umbrales is None:
        umbrales = np.arange(0.20, 0.81, 0.05)
    filas = []
    for u in umbrales:
        pred = (proba >= u).astype(int)
        filas.append({
            "umbral": round(float(u), 2),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "marcados": int(pred.sum()),
        })
    return pd.DataFrame(filas).round(3)


def beneficio_neto(y, proba, umbral, costo_oferta, valor_cliente, tasa_retencion):
    """Beneficio esperado de contactar a los clientes con proba >= umbral.

    - costo_oferta: lo que cuesta contactar/ofrecer algo a un cliente.
    - valor_cliente: lo que se pierde si el cliente se va.
    - tasa_retencion: fracción de clientes en riesgo que la oferta logra retener.
    """
    y = np.asarray(y)
    marcados = proba >= umbral
    verdaderos_positivos = (marcados & (y == 1)).sum()
    ahorro = verdaderos_positivos * tasa_retencion * valor_cliente
    gasto = marcados.sum() * costo_oferta
    return float(ahorro - gasto)