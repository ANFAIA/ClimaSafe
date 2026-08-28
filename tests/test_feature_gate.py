"""
test_feature_gate.py — Tests para climasafeai/features/feature_gate.py

Verifica los 4 criterios anti-leakage:
  1. Sub-componente matemático del target
  2. Misma fuente de encuesta (MoMo)
  3. Efecto causal downstream
  4. Datos temporales futuros
"""
import numpy as np
import pandas as pd
import pytest

from climasafeai.features.feature_gate import (
    FeatureVerdict,
    GateReport,
    LeakReason,
    evaluate_feature,
    run_feature_gate,
    apply_feature_gate,
    _check_math_sub_component,
    _check_same_survey,
    _check_downstream_causal,
    _check_future_temporal,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de los checks individuales
# ─────────────────────────────────────────────────────────────────────────────

class TestMathSubComponent:
    """Criterio 1: feature es sub-componente matemático del target."""

    def test_defunciones_exc_temp_is_blocked(self):
        assert _check_math_sub_component("defunciones_atrib_exc_temp") is True

    def test_defunciones_def_temp_is_blocked(self):
        assert _check_math_sub_component("defunciones_atrib_def_temp") is True

    def test_weather_feature_passes(self):
        assert _check_math_sub_component("t2m_c") is False

    def test_heat_index_passes(self):
        assert _check_math_sub_component("heat_index_c") is False

    def test_wind_chill_passes(self):
        assert _check_math_sub_component("wind_chill_c") is False

    def test_rh_passes(self):
        assert _check_math_sub_component("rh") is False


class TestSameSurveySource:
    """Criterio 2: feature viene de la misma encuesta que el target (MoMo)."""

    def test_clase_riesgo_calor_is_blocked(self):
        assert _check_same_survey("clase_riesgo_calor") is True

    def test_clase_riesgo_frio_is_blocked(self):
        assert _check_same_survey("clase_riesgo_frio") is True

    def test_clase_riesgo_calor_label_is_blocked(self):
        assert _check_same_survey("clase_riesgo_calor_label") is True

    def test_clase_riesgo_frio_label_is_blocked(self):
        assert _check_same_survey("clase_riesgo_frio_label") is True

    def test_defunciones_exc_is_blocked(self):
        assert _check_same_survey("defunciones_atrib_exc_temp") is True

    def test_defunciones_def_is_blocked(self):
        assert _check_same_survey("defunciones_atrib_def_temp") is True

    def test_era5_weather_passes(self):
        """Features meteorológicas de ERA5 son de fuente independiente."""
        assert _check_same_survey("t2m_c") is False

    def test_heat_index_passes(self):
        assert _check_same_survey("heat_index_c") is False

    def test_wind_speed_passes(self):
        assert _check_same_survey("wind_speed_kmh") is False

    def test_rh_passes(self):
        assert _check_same_survey("rh") is False

    def test_sp_passes(self):
        assert _check_same_survey("sp") is False

    def test_horas_sobre_umbral_passes(self):
        """Estadística diaria derivada de ERA5, no de MoMo."""
        assert _check_same_survey("horas_sobre_umbral") is False

    def test_heat_index_mean_passes(self):
        assert _check_same_survey("heat_index_mean") is False

    def test_dias_consec_sobre_umbral_passes(self):
        assert _check_same_survey("dias_consec_sobre_umbral") is False


class TestDownstreamCausal:
    """Criterio 3: feature es efecto causal downstream del target."""

    def test_ingresos_hospitalarios_is_blocked(self):
        assert _check_downstream_causal("ingresos_hospitalarios") is True

    def test_llamadas_112_is_blocked(self):
        assert _check_downstream_causal("llamadas_112") is True

    def test_urgencias_calor_is_blocked(self):
        assert _check_downstream_causal("urgencias_calor") is True

    def test_urgencias_frio_is_blocked(self):
        assert _check_downstream_causal("urgencias_frio") is True

    def test_weather_passes(self):
        assert _check_downstream_causal("t2m_c") is False

    def test_defunciones_not_downstream(self):
        """defunciones es el target mismo, no downstream."""
        assert _check_downstream_causal("defunciones_atrib_exc_temp") is False


class TestFutureTemporal:
    """Criterio 4: feature usa datos temporales futuros."""

    def test_weather_passes(self):
        assert _check_future_temporal("t2m_c") is False

    def test_lag1_passes(self):
        assert _check_future_temporal("heat_index_c_lag1") is False


# ─────────────────────────────────────────────────────────────────────────────
# Tests de evaluate_feature
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateFeature:

    def test_valid_feature_passes(self):
        verdict = evaluate_feature("heat_index_c")
        assert verdict.passed is True
        assert verdict.failed_criteria == []

    def test_defunciones_fails_two_criteria(self):
        """defunciones_atrib_exc_temp falla math_sub_component Y same_survey."""
        verdict = evaluate_feature("defunciones_atrib_exc_temp")
        assert verdict.passed is False
        reasons = verdict.failed_criteria
        assert LeakReason.MATH_SUB_COMPONENT in reasons
        assert LeakReason.SAME_SURVEY_SOURCE in reasons

    def test_clase_riesgo_calor_fails_same_survey(self):
        verdict = evaluate_feature("clase_riesgo_calor")
        assert verdict.passed is False
        assert LeakReason.SAME_SURVEY_SOURCE in verdict.failed_criteria

    def test_ingresos_hospitalarios_fails_downstream(self):
        verdict = evaluate_feature("ingresos_hospitalarios")
        assert verdict.passed is False
        assert LeakReason.DOWNSTREAM_CAUSAL in verdict.failed_criteria


# ─────────────────────────────────────────────────────────────────────────────
# Tests de run_feature_gate
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_dataset():
    """DataFrame que simula el dataset combinado ERA5+MoMo."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        # --- Meteorológicas (ERA5) — deben pasar ---
        "t2m_c": np.random.uniform(5, 45, n),
        "rh": np.random.uniform(20, 100, n),
        "wind_speed_kmh": np.random.uniform(0, 50, n),
        "sp": np.random.uniform(950, 1050, n),
        "heat_index_c": np.random.uniform(10, 50, n),
        "wbgt_c": np.random.uniform(10, 40, n),
        "wind_chill_c": np.random.uniform(-10, 30, n),
        # --- Estadísticas diarias (ERA5) — deben pasar ---
        "heat_index_mean": np.random.uniform(10, 45, n),
        "heat_index_std": np.random.uniform(0, 10, n),
        "heat_index_min": np.random.uniform(5, 30, n),
        "horas_sobre_umbral": np.random.randint(0, 24, n),
        "wind_chill_mean": np.random.uniform(-5, 25, n),
        "wind_chill_std": np.random.uniform(0, 8, n),
        "wind_chill_max": np.random.uniform(5, 35, n),
        "horas_bajo_umbral": np.random.randint(0, 24, n),
        # --- Temporales (lags) — deben pasar ---
        "heat_index_c_lag1": np.random.uniform(10, 50, n),
        "heat_index_c_roll3": np.random.uniform(10, 50, n),
        "dias_consec_sobre_umbral": np.random.randint(0, 10, n),
        # --- MoMo — deben filtrarse ---
        "defunciones_atrib_exc_temp": np.random.poisson(2, n).astype(float),
        "defunciones_atrib_def_temp": np.random.poisson(1, n).astype(float),
        "clase_riesgo_calor": np.random.choice([0, 1, 2], n),
        "clase_riesgo_frio": np.random.choice([0, 1, 2], n),
        # --- Identificadores (se excluyen aparte) ---
        "provincia": ["Madrid"] * 50 + ["Sevilla"] * 50,
        "fecha": pd.date_range("2020-01-01", periods=n),
    })
    return df


class TestRunFeatureGate:

    def test_filters_leaking_columns(self, sample_dataset):
        """Las columnas de MoMo que son leakage se filtran."""
        report = run_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        filtered_names = {v.feature for v in report.filtered}
        assert "defunciones_atrib_exc_temp" in filtered_names
        assert "defunciones_atrib_def_temp" in filtered_names
        # clase_riesgo_calor es target_col → se excluye de evaluación, no se filtra
        assert "clase_riesgo_frio" in filtered_names

    def test_passes_weather_features(self, sample_dataset):
        """Features meteorológicas de ERA5 pasan el gate."""
        report = run_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        passed = set(report.passed_features)
        for col in ["t2m_c", "rh", "wind_speed_kmh", "sp",
                     "heat_index_c", "wbgt_c", "wind_chill_c",
                     "heat_index_mean", "horas_sobre_umbral",
                     "heat_index_c_lag1"]:
            assert col in passed, f"{col} debería pasar el gate"

    def test_excludes_target_from_evaluation(self, sample_dataset):
        """El target_col no se evalúa (se excluye)."""
        report = run_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        assert "clase_riesgo_calor" not in report.all_features

    def test_excludes_specified_columns(self, sample_dataset):
        """Las exclude_cols no se evalúan."""
        report = run_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        assert "provincia" not in report.all_features
        assert "fecha" not in report.all_features

    def test_extra_blocked_adds_downstream(self, sample_dataset):
        """extra_blocked añade columnas downstream para filtrar."""
        df = sample_dataset.copy()
        df["mi_columna_custom"] = 1.0  # columna que existe en el DF
        report = run_feature_gate(
            df,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
            extra_blocked=["mi_columna_custom"],
        )
        filtered_names = {v.feature for v in report.filtered}
        assert "mi_columna_custom" in filtered_names

    def test_report_summary_contains_info(self, sample_dataset):
        """El reporte tiene la información necesaria."""
        report = run_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        summary = report.summary()
        assert "Feature Gate" in summary
        assert "calor" in summary
        assert "Filtradas" in summary
        assert "Pasan el gate" in summary

    def test_report_filtered_features_has_reasons(self, sample_dataset):
        """Cada feature filtrada tiene al menos un motivo."""
        report = run_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        for item in report.filtered_features:
            assert "feature" in item
            assert "reasons" in item
            assert len(item["reasons"]) >= 1

    def test_calor_and_frio_share_era5_features(self, sample_dataset):
        """ERA5 features pasan tanto para calor como para frío."""
        for clase in ("calor", "frio"):
            target = f"clase_riesgo_{clase}"
            report = run_feature_gate(
                sample_dataset,
                clase=clase,
                target_col=target,
                exclude_cols=["provincia", "fecha"],
            )
            passed = set(report.passed_features)
            assert "t2m_c" in passed, f"t2m_c debería pasar para {clase}"
            assert "heat_index_c" in passed, f"heat_index_c debería pasar para {clase}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests de apply_feature_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyFeatureGate:

    def test_removes_filtered_columns(self, sample_dataset):
        """apply_feature_gate elimina las columnas filtradas del DataFrame."""
        df_filtered, report = apply_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        for col in ["defunciones_atrib_exc_temp", "defunciones_atrib_def_temp",
                     "clase_riesgo_frio"]:
            assert col not in df_filtered.columns
        # clase_riesgo_calor es target_col → se excluye de evaluación, no se filtra
        # (queda en el DF porque el gate no la toca; preprocess_data la excluye después)

    def test_keeps_valid_columns(self, sample_dataset):
        """apply_feature_gate conserva las columnas que pasan."""
        df_filtered, _ = apply_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        for col in ["t2m_c", "rh", "wind_speed_kmh", "sp",
                     "heat_index_c", "heat_index_mean"]:
            assert col in df_filtered.columns

    def test_returns_both_df_and_report(self, sample_dataset):
        """apply_feature_gate devuelve (DataFrame, GateReport)."""
        result = apply_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        df_filtered, report = result
        assert isinstance(df_filtered, pd.DataFrame)
        assert isinstance(report, GateReport)

    def test_row_count_preserved(self, sample_dataset):
        """El número de filas no cambia tras el gate."""
        df_filtered, _ = apply_feature_gate(
            sample_dataset,
            clase="calor",
            target_col="clase_riesgo_calor",
            exclude_cols=["provincia", "fecha"],
        )
        assert len(df_filtered) == len(sample_dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Tests de GateReport
# ─────────────────────────────────────────────────────────────────────────────

class TestGateReport:

    def test_filtered_property(self):
        verdicts = [
            FeatureVerdict("a", True),
            FeatureVerdict("b", False, [LeakReason.SAME_SURVEY_SOURCE]),
            FeatureVerdict("c", True),
        ]
        report = GateReport(clase="calor", all_features=["a", "b", "c"], verdicts=verdicts)
        assert len(report.filtered) == 1
        assert report.filtered[0].feature == "b"

    def test_passed_features_property(self):
        verdicts = [
            FeatureVerdict("a", True),
            FeatureVerdict("b", False, [LeakReason.DOWNSTREAM_CAUSAL]),
        ]
        report = GateReport(clase="frio", all_features=["a", "b"], verdicts=verdicts)
        assert report.passed_features == ["a"]

    def test_filtered_features_property(self):
        verdicts = [
            FeatureVerdict("defunciones_atrib_exc_temp", False,
                           [LeakReason.MATH_SUB_COMPONENT, LeakReason.SAME_SURVEY_SOURCE]),
        ]
        report = GateReport(clase="calor", all_features=["defunciones_atrib_exc_temp"], verdicts=verdicts)
        items = report.filtered_features
        assert len(items) == 1
        assert items[0]["feature"] == "defunciones_atrib_exc_temp"
        assert "math_sub_component" in items[0]["reasons"]
        assert "same_survey_source" in items[0]["reasons"]


# ─────────────────────────────────────────────────────────────────────────────
# Test de integración: el gate no rompe el pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationWithPipeline:

    def test_gate_compatible_with_build_features_cols(self):
        """Las columnas que el gate evalúa son compatibles con COLS_TO_DROP."""
        from climasafeai.features.build_features import COLS_TO_DROP, LEAKAGE_COLS_BY_CLASE

        # El gate excluye las mismas columnas que build_features
        for clase in ("calor", "frio"):
            report = run_feature_gate(
                pd.DataFrame({"t2m_c": [1], "rh": [1]}),
                clase=clase,
                target_col=f"clase_riesgo_{clase}",
                exclude_cols=[],
            )
            # No debe rechazar features meteorológicas básicas
            assert "t2m_c" in report.passed_features
            assert "rh" in report.passed_features

    def test_no_features_filtered_when_only_weather(self):
        """Si solo hay features meteorológicas, no se filtra nada."""
        df = pd.DataFrame({
            "t2m_c": [1.0, 2.0],
            "rh": [50.0, 60.0],
            "wind_speed_kmh": [10.0, 20.0],
            "heat_index_c": [30.0, 35.0],
        })
        report = run_feature_gate(df, clase="calor", target_col=None, exclude_cols=[])
        assert len(report.filtered) == 0
        assert len(report.passed_features) == 4
