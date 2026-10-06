import pandas as pd
import pytest

from src.data import limpiar_datos


def test_totalcharges_queda_numerico_sin_nulos(df_crudo):
    limpio = limpiar_datos(df_crudo)
    assert pd.api.types.is_float_dtype(limpio["TotalCharges"])
    assert limpio["TotalCharges"].isna().sum() == 0


def test_clientes_nuevos_quedan_con_totalcharges_cero(df_crudo):
    limpio = limpiar_datos(df_crudo)
    nuevos = limpio["tenure"] == 0
    assert nuevos.any()
    assert (limpio.loc[nuevos, "TotalCharges"] == 0).all()


def test_se_eliminan_las_comillas_literales(df_crudo):
    limpio = limpiar_datos(df_crudo)
    for col in limpio.select_dtypes(exclude="number").columns:
        assert not limpio[col].str.contains("'").any(), f"quedan comillas en {col}"
    assert set(limpio["Contract"]) == {"Month-to-month", "One year", "Two year"}


def test_churn_queda_binario(df_crudo):
    limpio = limpiar_datos(df_crudo)
    assert set(limpio["Churn"].unique()) <= {0, 1}
    assert limpio["Churn"].sum() == (df_crudo["Churn"] == "Yes").sum()


def test_no_pierde_ni_agrega_filas_ni_columnas(df_crudo):
    limpio = limpiar_datos(df_crudo)
    assert limpio.shape == df_crudo.shape


def test_no_modifica_el_dataframe_original(df_crudo):
    copia = df_crudo.copy()
    limpiar_datos(df_crudo)
    pd.testing.assert_frame_equal(df_crudo, copia)


def test_falla_si_hay_totalcharges_vacio_con_tenure_positivo(df_crudo):
    fila = df_crudo.index[df_crudo["tenure"] > 0][0]
    df_crudo.loc[fila, "TotalCharges"] = " "
    with pytest.raises(ValueError):
        limpiar_datos(df_crudo)