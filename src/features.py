"""Ingeniería de variables y preprocesamiento."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"

COLS_ADICIONALES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def crear_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simplifica categorías y crea variables nuevas. No modifica el original."""
    df = df.copy()

    # 1. "No internet service" / "No phone service" -> "No"
    for col in COLS_ADICIONALES + ["MultipleLines"]:
        df[col] = df[col].replace(
            {"No internet service": "No", "No phone service": "No"}
        )

    # 2. Variables nuevas, calculadas fila por fila
    df["tiene_internet"] = (df["InternetService"] != "No").astype(int)
    df["num_servicios"] = (df[COLS_ADICIONALES] == "Yes").sum(axis=1)
    df["contrato_mensual"] = (df["Contract"] == "Month-to-month").astype(int)
    df["pago_electronico"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["tenure_grupo"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    ).astype(str)

    return df


def separar_X_y(df: pd.DataFrame):
    """Separa las variables predictoras (X) de la variable objetivo (y)."""
    return df.drop(columns=TARGET), df[TARGET]


def construir_preprocesador(X: pd.DataFrame) -> ColumnTransformer:
    """Escala las numéricas y convierte las categóricas en columnas 0/1."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])