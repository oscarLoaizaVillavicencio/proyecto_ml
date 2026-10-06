"""Datos sintéticos compartidos por los tests.

Los tests NO dependen de data/raw (que no se sube a GitHub): construyen un
DataFrame pequeño que imita el formato crudo del dataset, con sus defectos
(comillas literales y TotalCharges vacío en clientes con tenure = 0).
"""
import numpy as np
import pandas as pd
import pytest


def construir_df_crudo(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contrato = rng.choice(["Month-to-month", "'One year'", "'Two year'"], n, p=[0.5, 0.25, 0.25])
    internet = rng.choice(["DSL", "'Fiber optic'", "No"], n)
    sin_internet = internet == "No"
    telefono = rng.choice(["Yes", "No"], n)

    def servicio_extra():
        valores = rng.choice(["Yes", "No"], n)
        return np.where(sin_internet, "'No internet service'", valores)

    tenure = rng.integers(0, 73, n)
    tenure[:3] = 0  # garantiza clientes nuevos
    mensual = np.round(rng.uniform(18, 119, n), 2)
    total = np.where(tenure == 0, " ", (tenure * mensual).round(2).astype(str))

    # El churn depende del contrato mensual, para que el modelo tenga algo que aprender
    prob_churn = np.where(contrato == "Month-to-month", 0.6, 0.1)
    churn = np.where(rng.random(n) < prob_churn, "Yes", "No")

    return pd.DataFrame({
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": telefono,
        "MultipleLines": np.where(telefono == "No", "'No phone service'", rng.choice(["Yes", "No"], n)),
        "InternetService": internet,
        "OnlineSecurity": servicio_extra(),
        "OnlineBackup": servicio_extra(),
        "DeviceProtection": servicio_extra(),
        "TechSupport": servicio_extra(),
        "StreamingTV": servicio_extra(),
        "StreamingMovies": servicio_extra(),
        "Contract": contrato,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["'Electronic check'", "'Mailed check'", "'Bank transfer (automatic)'"], n),
        "MonthlyCharges": mensual,
        "TotalCharges": total,
        "Churn": churn,
    })


@pytest.fixture
def df_crudo() -> pd.DataFrame:
    return construir_df_crudo()


@pytest.fixture
def df_limpio(df_crudo) -> pd.DataFrame:
    from src.data import limpiar_datos
    return limpiar_datos(df_crudo)