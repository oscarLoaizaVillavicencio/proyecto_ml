"""Funciones para obtener y cargar los datos del proyecto."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAIZ = Path(__file__).resolve().parents[1]
RAW_PATH = RAIZ / "data" / "raw" / "telco_churn.csv"


def descargar_datos(forzar: bool = False) -> Path:
    """Descarga el dataset Telco Churn desde OpenML y lo guarda en data/raw."""
    if RAW_PATH.exists() and not forzar:
        print(f"El archivo ya existe: {RAW_PATH}")
        return RAW_PATH

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_openml(data_id=42178, as_frame=True).frame
    df.to_csv(RAW_PATH, index=False)
    print(f"Datos guardados en {RAW_PATH} con forma {df.shape}")
    return RAW_PATH


def cargar_raw() -> pd.DataFrame:
    """Carga el CSV crudo sin modificarlo."""
    return pd.read_csv(RAW_PATH)


if __name__ == "__main__":
    descargar_datos()