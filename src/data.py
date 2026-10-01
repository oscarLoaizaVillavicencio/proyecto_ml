"""Funciones para obtener, limpiar y guardar los datos del proyecto."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

RAIZ = Path(__file__).resolve().parents[1]
RAW_PATH = RAIZ / "data" / "raw" / "telco_churn.csv"
PROCESSED_PATH = RAIZ / "data" / "processed" / "telco_churn_limpio.csv"


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


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset crudo y devuelve una copia lista para modelar."""
    df = df.copy()

    # 1. Quitar espacios y comillas literales en las columnas de texto
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].str.strip().str.strip("'").str.strip()

    # 2. TotalCharges: texto -> número. Los vacíos son clientes con tenure = 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    sin_cobro = df["TotalCharges"].isna()
    if not (df.loc[sin_cobro, "tenure"] == 0).all():
        raise ValueError("Hay TotalCharges vacíos con tenure > 0: revisar los datos")
    df.loc[sin_cobro, "TotalCharges"] = 0.0

    # 3. Variable objetivo en binario: 1 = abandonó, 0 = se quedó
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def guardar_procesado(df: pd.DataFrame) -> Path:
    """Guarda el dataset limpio en data/processed."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    return PROCESSED_PATH


def cargar_procesado() -> pd.DataFrame:
    """Carga el dataset limpio."""
    return pd.read_csv(PROCESSED_PATH)


if __name__ == "__main__":
    descargar_datos()