"""Preparación de datos, entrenamiento y validación de modelos."""
import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data import RAIZ, cargar_procesado
from src.features import construir_preprocesador, crear_features, separar_X_y

MODELS_DIR = RAIZ / "models"
RANDOM_STATE = 42


def preparar_datos(test_size: float = 0.2):
    """Crea las variables y divide en entrenamiento y prueba."""
    df = crear_features(cargar_procesado())
    X, y = separar_X_y(df)
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


def modelos_candidatos() -> dict:
    """Modelos que vamos a comparar."""
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "regresion_logistica": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def construir_pipeline(estimador, X) -> Pipeline:
    """Pipeline sin entrenar: preprocesador + modelo."""
    return Pipeline([
        ("prep", construir_preprocesador(X)),
        ("modelo", estimador),
    ])


def entrenar(estimador, X_train, y_train) -> Pipeline:
    """Entrena preprocesador + modelo juntos, usando solo datos de entrenamiento."""
    return construir_pipeline(estimador, X_train).fit(X_train, y_train)


def _cv(k: int = 5) -> StratifiedKFold:
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)


def validar_cruzado(estimadores: dict, X_train, y_train, k: int = 5) -> pd.DataFrame:
    """Validación cruzada de varios modelos. Devuelve media y desviación por métrica."""
    metricas = ["roc_auc", "recall", "precision", "f1"]
    filas = {}
    for nombre, est in estimadores.items():
        res = cross_validate(
            construir_pipeline(est, X_train), X_train, y_train,
            cv=_cv(k), scoring=metricas, n_jobs=-1,
        )
        fila = {}
        for m in metricas:
            fila[f"{m}_media"] = res[f"test_{m}"].mean()
            fila[f"{m}_std"] = res[f"test_{m}"].std()
        filas[nombre] = fila
    return pd.DataFrame(filas).T.round(3)


def probabilidades_oof(estimador, X_train, y_train, k: int = 5):
    """Probabilidades de churn para cada fila de entrenamiento, calculadas
    por un modelo que NO vio esa fila (out-of-fold)."""
    return cross_val_predict(
        construir_pipeline(estimador, X_train), X_train, y_train,
        cv=_cv(k), method="predict_proba", n_jobs=-1,
    )[:, 1]


def guardar_modelo(modelo: Pipeline, nombre: str):
    """Guarda el modelo entrenado en la carpeta models."""
    MODELS_DIR.mkdir(exist_ok=True)
    ruta = MODELS_DIR / f"{nombre}.joblib"
    joblib.dump(modelo, ruta)
    return ruta