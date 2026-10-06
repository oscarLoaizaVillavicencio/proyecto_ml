import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src import train
from src.evaluate import beneficio_neto, evaluar, matriz_confusion, tabla_umbrales
from src.features import crear_features, separar_X_y
from src.train import entrenar


@pytest.fixture
def particion(df_limpio):
    X, y = separar_X_y(crear_features(df_limpio))
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)


@pytest.fixture
def modelo(particion):
    X_train, _, y_train, _ = particion
    return entrenar(LogisticRegression(max_iter=1000, class_weight="balanced"), X_train, y_train)


def test_probabilidades_validas(modelo, particion):
    _, X_test, _, _ = particion
    proba = modelo.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1)


def test_el_modelo_supera_al_azar(modelo, particion):
    _, X_test, _, y_test = particion
    assert evaluar(modelo, X_test, y_test)["roc_auc"] > 0.65


def test_el_baseline_no_detecta_ningun_abandono(particion):
    X_train, X_test, y_train, y_test = particion
    baseline = entrenar(DummyClassifier(strategy="most_frequent"), X_train, y_train)
    metricas = evaluar(baseline, X_test, y_test)
    assert metricas["recall"] == 0
    assert metricas["roc_auc"] == 0.5


def test_acepta_categorias_nuevas_sin_fallar(modelo, particion):
    _, X_test, _, _ = particion
    fila = X_test.head(1).copy()
    fila["PaymentMethod"] = "Criptomoneda"
    assert modelo.predict_proba(fila).shape == (1, 2)


def test_el_pipeline_no_necesita_preprocesar_a_mano(modelo, particion):
    """El Pipeline recibe las columnas crudas (con texto) y las procesa solo."""
    _, X_test, _, _ = particion
    assert len(modelo.predict(X_test)) == len(X_test)


def test_matriz_de_confusion_suma_el_total(modelo, particion):
    _, X_test, _, y_test = particion
    assert matriz_confusion(modelo, X_test, y_test).to_numpy().sum() == len(y_test)


def test_bajar_el_umbral_sube_el_recall(modelo, particion):
    _, X_test, _, y_test = particion
    alto = evaluar(modelo, X_test, y_test, umbral=0.7)["recall"]
    bajo = evaluar(modelo, X_test, y_test, umbral=0.3)["recall"]
    assert bajo >= alto


def test_tabla_umbrales_marca_menos_clientes_al_subir_el_umbral():
    y = np.array([1, 0, 1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.6, 0.4, 0.3, 0.1])
    tabla = tabla_umbrales(y, proba, umbrales=[0.2, 0.5, 0.85])
    assert tabla["marcados"].tolist() == [5, 3, 1]


def test_beneficio_neto_calculado_a_mano():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    # umbral 0.5 marca 2 clientes (1 acierto): ahorro 1*0.5*300 = 150, gasto 2*20 = 40
    assert beneficio_neto(y, proba, 0.5, costo_oferta=20, valor_cliente=300, tasa_retencion=0.5) == 110


def test_guardar_y_cargar_modelo_da_las_mismas_predicciones(modelo, particion, tmp_path, monkeypatch):
    import joblib
    _, X_test, _, _ = particion
    monkeypatch.setattr(train, "MODELS_DIR", tmp_path)
    ruta = train.guardar_modelo(modelo, "modelo_prueba")
    assert ruta.exists()
    cargado = joblib.load(ruta)
    assert np.allclose(cargado.predict_proba(X_test), modelo.predict_proba(X_test))