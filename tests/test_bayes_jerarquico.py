"""
test_bayes_jerarquico.py — Tests del modelo jerárquico por provincia (BAYES-001).

Cubren: entrenamiento y predicción sobre datos sintéticos pequeños (sin
datasets grandes ni red), salida con intervalos de credibilidad, traducción a
las tres clases, comparativa contra el baseline del ensemble con su tabla, y
diagnósticos del muestreo (r_hat/ESS). Rápidos y deterministas (seeds fijas).
"""
import numpy as np
import pandas as pd
import pytest

from climasafeai.models.bayes_jerarquico import (
    CLASES,
    ModeloJerarquico,
    cargar_datos_entrenamiento,
    comparar_con_ensemble,
    comparativa_muchos_pocos_datos,
    ess,
    intervalo_credibilidad,
    particion_temporal_por_fecha,
    r_hat,
    traducir_a_clase,
)


def _datos_sinteticos(seed: int = 7, n_chica: int = 3) -> pd.DataFrame:
    """DataFrame sintético: 3 provincias (P0 pequeña, P1/P2 grandes), 2 features
    y clase ordinal 0/1/2 con relación clara a feat_a + efecto provincial.

    Las fechas se reparten por todo el rango (no por bloques) y P0/P1 fuerzan
    una fila en el último tramo, para que TODAS las provincias tengan filas en
    train y en test con cualquier test_size razonable."""
    rng = np.random.default_rng(seed)
    fechas_totales = pd.date_range("2020-01-01", periods=80, freq="D")
    filas = []
    for prov, n, efecto in [("P0", n_chica, 0.0), ("P1", 40, 1.0), ("P2", 37, -0.5)]:
        idx_fechas = np.sort(rng.choice(len(fechas_totales), size=n, replace=False))
        # garantizar presencia en test (último 20% de fechas)
        if prov == "P0":
            idx_fechas[0] = len(fechas_totales) - 1
        elif prov == "P1":
            idx_fechas[-1] = len(fechas_totales) - 2
        feat_a = rng.normal(size=n)
        feat_b = rng.normal(size=n)
        score = 1.5 * feat_a + 0.4 * feat_b + efecto
        y = np.where(score > 0.8, 2, np.where(score > -0.4, 1, 0))
        filas.append(pd.DataFrame({
            "fecha": fechas_totales[idx_fechas],
            "provincia": prov,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "clase": y,
        }))
    return pd.concat(filas, ignore_index=True).sort_values("fecha").reset_index(drop=True)


def test_importa_sin_pgmpy():
    """El módulo no depende de pgmpy (a diferencia de bayes.py)."""
    import climasafeai.models.bayes_jerarquico as bj

    assert "pgmpy" not in dir(bj)


def test_particion_temporal_no_solapa_y_coge_ultimas_fechas():
    df = _datos_sinteticos()
    df_train, df_test = particion_temporal_por_fecha(df, test_size=0.2)
    fechas_train = set(pd.to_datetime(df_train["fecha"]))
    fechas_test = set(pd.to_datetime(df_test["fecha"]))
    assert fechas_train.isdisjoint(fechas_test)
    assert max(fechas_train) < min(fechas_test)
    assert len(fechas_test) <= round(len(df["fecha"].unique()) * 0.2) + 1


def test_modelo_entrena_y_predice_sintetico():
    df = _datos_sinteticos()
    df_train, df_test = particion_temporal_por_fecha(df, test_size=0.25)
    modelo = ModeloJerarquico(clase="calor", features=["feat_a", "feat_b"], random_state=42)
    modelo.fit(df_train, col_y="clase", n_cadenas=2, n_warmup=150, n_muestras=100)
    pred = modelo.predecir(df_test)
    assert len(pred) == len(df_test)
    assert set(pred["clase_predicha"]).issubset({0, 1, 2})
    assert set(pred["clase_predicha_label"]).issubset(set(CLASES))
    assert pred["prob_riesgo_mediana"].between(0, 1).all()
    assert not pred[["prob_riesgo_mediana", "prob_riesgo_lower", "prob_riesgo_upper"]].isna().any().any()


def test_salida_tiene_intervalos():
    df = _datos_sinteticos()
    df_train, df_test = particion_temporal_por_fecha(df, test_size=0.25)
    modelo = ModeloJerarquico(clase="calor", features=["feat_a", "feat_b"], random_state=1)
    modelo.fit(df_train, col_y="clase", n_cadenas=2, n_warmup=150, n_muestras=100)
    pred = modelo.predecir(df_test, alpha=0.1)
    assert (pred["prob_riesgo_lower"] <= pred["prob_riesgo_mediana"]).all()
    assert (pred["prob_riesgo_mediana"] <= pred["prob_riesgo_upper"]).all()
    for col in ("prob0_mediana", "prob1_mediana", "prob2_mediana"):
        assert pred[col].between(0, 1).all()
    # Cada muestra posterior suma 1 (por construcción), pero las medianas por
    # clase se calculan por separado y mediana(p1)+mediana(p2) puede diferir de
    # mediana(p1+p2) en filas con distribuciones asimétricas. La identidad
    # exacta es entre medias: el sesgo medio del artefacto debe ser pequeño.
    assert abs(float(
        (pred["prob1_mediana"] + pred["prob2_mediana"] - pred["prob_riesgo_mediana"]).mean()
    )) < 0.05
    assert abs(float((pred["prob0_mediana"] + pred["prob1_mediana"] + pred["prob2_mediana"]).mean()) - 1.0) < 0.05


def test_traduccion_a_clases_usa_cascada_del_pipeline():
    """Traduce probs (n,3) con la cascada P(2)>=t2 → PELIGRO; P(1)+P(2)>=t1 → PRECAUCIÓN."""
    probs = np.array([
        [0.90, 0.08, 0.02],   # p2<t2 y riesgo< t1  → SEGURO
        [0.70, 0.28, 0.02],   # p2<t2 y riesgo>=t1  → PRECAUCIÓN
        [0.30, 0.30, 0.40],   # p2>=t2              → PELIGRO
        [0.95, 0.04, 0.01],   # SEGURO
    ])
    clases = traducir_a_clase(probs, clase="calor")
    assert list(clases) == [0, 1, 2, 0]
    # coherencia con apply_class_thresholds del pipeline
    from climasafeai.models.predict_model import apply_class_thresholds

    esperado = apply_class_thresholds(
        probs, t1=0.25, t2=0.10
    )
    assert list(clases) == list(esperado)
    assert CLASES[clases[2]] == "PELIGRO"
    assert CLASES[clases[1]] == "PRECAUCION"


def test_resumen_muestreo_tiene_metricas():
    df = _datos_sinteticos()
    df_train, _ = particion_temporal_por_fecha(df, test_size=0.25)
    modelo = ModeloJerarquico(clase="calor", features=["feat_a", "feat_b"], random_state=3)
    modelo.fit(df_train, col_y="clase", n_cadenas=2, n_warmup=120, n_muestras=80)
    resumen = modelo.resumen_muestreo()
    for col in ("parametro", "mediana", "r_hat", "ess", "aceptacion", "divergencias"):
        assert col in resumen.columns
    assert np.isfinite(resumen["r_hat"]).all()
    assert np.isfinite(resumen["ess"]).all()
    assert (resumen["ess"] > 0).all()
    assert (resumen["divergencias"] == 0).all()
    # parámetros por provincia: beta_x2 + cortes + u_j + tau
    assert "u_P0" in set(resumen["parametro"])
    assert "tau" in set(resumen["parametro"])


def test_comparativa_con_ensemble_produce_tabla():
    df = _datos_sinteticos()
    comp = comparar_con_ensemble(
        df, clase="calor", features=["feat_a", "feat_b"], col_y="clase",
        test_size=0.25, n_cadenas=2, n_warmup=120, n_muestras=80, random_state=5,
    )
    tabla = comp["tabla"]
    for col in ("provincia", "n_test", "acc_baseline", "f1_baseline", "brier_baseline",
                "acc_jerarquico", "f1_jerarquico", "brier_jerarquico",
                "cobertura_90", "ganancia_f1"):
        assert col in tabla.columns
    # una fila por provincia + agregado TOTAL
    assert set(tabla["provincia"]) == {"P0", "P1", "P2", "TOTAL"}
    for col in ("acc_baseline", "f1_baseline", "brier_baseline",
                "acc_jerarquico", "f1_jerarquico", "brier_jerarquico", "cobertura_90"):
        assert tabla[col].between(0, 1).all()
    total = tabla[tabla["provincia"] == "TOTAL"].iloc[0]
    assert total["n_test"] == comp["prediccion"].shape[0]
    assert isinstance(comp["modelo"], ModeloJerarquico)


def test_provincia_pocos_datos_tiene_mas_incertidumbre():
    """La razón de ser del partial pooling: el efecto u de P0 (3 filas) debe
    tener un intervalo de credibilidad MÁS ANCHO que el de P1 (40 filas) —
    menos datos propios, más incertidumbre, sin sobreajustar. Se mide sobre la
    traza del efecto provincial (u_j), no sobre el intervalo de prob_riesgo
    (que depende de la zona de la sigmoide donde caiga el score)."""
    df = _datos_sinteticos()
    df_train, df_test = particion_temporal_por_fecha(df, test_size=0.25)
    modelo = ModeloJerarquico(clase="calor", features=["feat_a", "feat_b"], random_state=11)
    modelo.fit(df_train, col_y="clase", n_cadenas=2, n_warmup=150, n_muestras=100)
    u_p0 = np.concatenate([c["u"][:, 0] for c in modelo._cadenas])
    u_p1 = np.concatenate([c["u"][:, 1] for c in modelo._cadenas])
    ancho_p0 = np.percentile(u_p0, 97.5) - np.percentile(u_p0, 2.5)
    ancho_p1 = np.percentile(u_p1, 97.5) - np.percentile(u_p1, 2.5)
    assert ancho_p0 > ancho_p1
    # y el efecto estimado de P1 (efecto real +1.0) no se colapsa a 0
    assert float(np.mean(u_p1)) > 0.2


def test_comparativa_muchos_pocos_datos():
    df = _datos_sinteticos()
    comp = comparar_con_ensemble(
        df, clase="calor", features=["feat_a", "feat_b"], col_y="clase",
        test_size=0.25, n_cadenas=2, n_warmup=100, n_muestras=60, random_state=5,
    )
    tabla = comp["tabla"]
    n_episodios = df[df["clase"] > 0].groupby("provincia").size().to_dict()
    comparativa = comparativa_muchos_pocos_datos(tabla, n_episodios)
    for col in ("grupo", "n_provincias", "episodios_medio", "f1_baseline_medio",
                "f1_jerarquico_medio", "cobertura_90_media", "ganancia_f1_media"):
        assert col in comparativa.columns
    assert set(comparativa["grupo"]) == {"pocos_datos", "muchos_datos"}
    # P0 (3 filas) cae en pocos_datos
    pocos = comparativa[comparativa["grupo"] == "pocos_datos"].iloc[0]
    assert pocos["episodios_medio"] <= 3


def test_intervalo_credibilidad_y_rhat_ess_basicos():
    rng = np.random.default_rng(0)
    traza = rng.normal(size=500)
    lo, hi = intervalo_credibilidad(traza, alpha=0.1)
    assert lo < 0 < hi  # centrado en 0
    assert abs(lo) < 2.5 and abs(hi) < 2.5
    # r_hat ~ 1 cuando las cadenas comparten distribución
    cadenas = np.stack([rng.normal(size=300) for _ in range(3)])
    assert abs(r_hat(cadenas) - 1.0) < 0.2
    assert 0 < ess(traza) <= 500
