import numpy as np

from src.features import (
    COLS_ADICIONALES, construir_preprocesador, crear_features, separar_X_y,
)


def test_se_crean_las_variables_nuevas(df_limpio):
    out = crear_features(df_limpio)
    nuevas = {"tiene_internet", "num_servicios", "contrato_mensual",
              "pago_electronico", "tenure_grupo"}
    assert nuevas <= set(out.columns)


def test_ya_no_quedan_categorias_redundantes(df_limpio):
    out = crear_features(df_limpio)
    for col in COLS_ADICIONALES + ["MultipleLines"]:
        assert not out[col].isin(["No internet service", "No phone service"]).any()


def test_tiene_internet_y_num_servicios(df_limpio):
    out = crear_features(df_limpio)
    assert (out["tiene_internet"] == (out["InternetService"] != "No")).all()
    assert out["num_servicios"].between(0, len(COLS_ADICIONALES)).all()
    # Sin internet no puede haber servicios extra
    assert (out.loc[out["tiene_internet"] == 0, "num_servicios"] == 0).all()


def test_contrato_mensual_coincide_con_contract(df_limpio):
    out = crear_features(df_limpio)
    assert (out["contrato_mensual"] == (out["Contract"] == "Month-to-month")).all()


def test_tenure_grupo_respeta_los_limites(df_limpio):
    base = df_limpio.head(5).copy()
    base["tenure"] = [0, 6, 7, 48, 72]
    out = crear_features(base)
    assert out["tenure_grupo"].tolist() == ["0-6", "0-6", "7-12", "25-48", "49-72"]


def test_no_modifica_el_dataframe_original(df_limpio):
    columnas_antes = list(df_limpio.columns)
    crear_features(df_limpio)
    assert list(df_limpio.columns) == columnas_antes


def test_separar_X_y_no_deja_el_objetivo_en_X(df_limpio):
    X, y = separar_X_y(crear_features(df_limpio))
    assert "Churn" not in X.columns
    assert len(X) == len(y)


def test_preprocesador_genera_matriz_sin_nulos(df_limpio):
    X, _ = separar_X_y(crear_features(df_limpio))
    matriz = construir_preprocesador(X).fit_transform(X)
    matriz = matriz.toarray() if hasattr(matriz, "toarray") else matriz
    assert matriz.shape[0] == len(X)
    assert matriz.shape[1] > X.shape[1]  # el one-hot expande las categóricas
    assert not np.isnan(matriz).any()