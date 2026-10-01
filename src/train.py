"""Preparación de datos y entrenamiento de modelos."""
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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


def entrenar(estimador, X_train, y_train) -> Pipeline:
    """Entrena preprocesador + modelo juntos, usando solo datos de entrenamiento."""
    pipeline = Pipeline([
        ("prep", construir_preprocesador(X_train)),
        ("modelo", estimador),
    ])
    return pipeline.fit(X_train, y_train)


def guardar_modelo(modelo: Pipeline, nombre: str):
    """Guarda el modelo entrenado en la carpeta models."""
    MODELS_DIR.mkdir(exist_ok=True)
    ruta = MODELS_DIR / f"{nombre}.joblib"
    joblib.dump(modelo, ruta)
    return ruta