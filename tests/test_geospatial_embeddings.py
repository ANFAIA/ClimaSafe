"""Tests for geospatial_embeddings module (ML-003)."""
import numpy as np
import pandas as pd
import pytest

from climasafeai.features.geospatial_embeddings import (
    CensusDemographicsEmbedding,
    SpatialCoordinateEmbedding,
    GeospatialEmbeddingProvider,
    merge_embeddings,
)

ALL_PROVIDERS = [CensusDemographicsEmbedding, SpatialCoordinateEmbedding]
N_PROVINCES = 45  # número de provincias en el dataset


class TestProviderInterface:
    """Cada provider debe cumplir la interfaz base."""

    @pytest.mark.parametrize("provider_cls", ALL_PROVIDERS)
    def test_is_subclass(self, provider_cls):
        assert issubclass(provider_cls, GeospatialEmbeddingProvider)

    @pytest.mark.parametrize("provider_cls", ALL_PROVIDERS)
    def test_has_name(self, provider_cls):
        p = provider_cls()
        assert isinstance(p.name, str) and len(p.name) > 0

    @pytest.mark.parametrize("provider_cls", ALL_PROVIDERS)
    def test_has_feature_names(self, provider_cls):
        p = provider_cls()
        assert isinstance(p.feature_names, list) and len(p.feature_names) > 0
        assert all(isinstance(f, str) for f in p.feature_names)

    @pytest.mark.parametrize("provider_cls", ALL_PROVIDERS)
    def test_n_features_matches(self, provider_cls):
        p = provider_cls()
        assert p.n_features == len(p.feature_names)


class TestCensusDemographicsEmbedding:
    """Tests del provider demográfico (census)."""

    def setup_method(self):
        self.provider = CensusDemographicsEmbedding()

    def test_returns_all_provinces(self):
        emb = self.provider.get_embeddings()
        assert "provincia" in emb.columns
        assert len(emb) >= N_PROVINCES

    def test_embedding_dimensions(self):
        emb = self.provider.get_embeddings()
        assert emb.shape[1] == 1 + self.provider.n_features  # provincia + features

    def test_all_features_numeric(self):
        emb = self.provider.get_embeddings()
        for col in self.provider.feature_names:
            assert pd.api.types.is_numeric_dtype(emb[col]), f"{col} no es numérica"

    def test_no_nan(self):
        emb = self.provider.get_embeddings()
        for col in self.provider.feature_names:
            assert emb[col].notna().all(), f"{col} tiene NaN"

    def test_values_in_reasonable_range(self):
        emb = self.provider.get_embeddings()
        assert (emb["pct_envejecimiento"] >= 0).all()
        assert (emb["pct_envejecimiento"] <= 100).all()
        assert (emb["pct_mujeres"] >= 40).all()
        assert (emb["pct_mujeres"] <= 60).all()
        assert (emb["log_poblacion"] > 0).all()


class TestSpatialCoordinateEmbedding:
    """Tests del provider espacial (coordenadas)."""

    def setup_method(self):
        self.provider = SpatialCoordinateEmbedding()

    def test_returns_all_provinces(self):
        emb = self.provider.get_embeddings()
        assert "provincia" in emb.columns
        assert len(emb) >= N_PROVINCES

    def test_embedding_dimensions(self):
        emb = self.provider.get_embeddings()
        assert emb.shape[1] == 1 + self.provider.n_features

    def test_all_features_numeric(self):
        emb = self.provider.get_embeddings()
        for col in self.provider.feature_names:
            assert pd.api.types.is_numeric_dtype(emb[col]), f"{col} no es numérica"

    def test_lat_lon_in_range(self):
        emb = self.provider.get_embeddings()
        assert (emb["lat_norm"] >= 0).all() and (emb["lat_norm"] <= 1).all()
        assert (emb["lon_norm"] >= 0).all() and (emb["lon_norm"] <= 1).all()

    def test_dist_madrid_positive(self):
        emb = self.provider.get_embeddings()
        assert (emb["dist_madrid_km"] >= 0).all()

    def test_coastal_binary(self):
        emb = self.provider.get_embeddings()
        assert set(emb["es_costera"].unique()).issubset({0.0, 1.0})


class TestMergeEmbeddings:
    """Tests de merge con el DataFrame de entrenamiento."""

    def _make_dummy_df(self) -> pd.DataFrame:
        """Crea un DataFrame dummy con las 45 provincias del dataset."""
        from climasafeai.features.external_features import _EMBEDDED_DEMOGRAPHICS
        return pd.DataFrame({
            "provincia": list(_EMBEDDED_DEMOGRAPHICS.keys()),
            "t2m_c": np.random.randn(N_PROVINCES),
            "fecha": "2023-01-01",
        })

    def test_merge_census(self):
        df = self._make_dummy_df()
        result = merge_embeddings(df, [CensusDemographicsEmbedding()])
        assert "census_pct_envejecimiento" in result.columns
        assert len(result) == N_PROVINCES

    def test_merge_spatial(self):
        df = self._make_dummy_df()
        result = merge_embeddings(df, [SpatialCoordinateEmbedding()])
        assert "spatial_lat_norm" in result.columns
        assert len(result) == N_PROVINCES

    def test_merge_both(self):
        df = self._make_dummy_df()
        result = merge_embeddings(df, [
            CensusDemographicsEmbedding(),
            SpatialCoordinateEmbedding(),
        ])
        assert "census_pct_envejecimiento" in result.columns
        assert "spatial_lat_norm" in result.columns
        # Columnas originales intactas
        assert "t2m_c" in result.columns
        assert "provincia" in result.columns

    def test_merge_preserves_rows(self):
        df = self._make_dummy_df()
        result = merge_embeddings(df, [CensusDemographicsEmbedding()])
        assert len(result) == len(df)

    def test_merge_with_real_dataset(self):
        """Merge con el dataset real (si existe)."""
        import os
        path = "data/processed/dataset_calor_labeled.parquet"
        if not os.path.exists(path):
            pytest.skip("Dataset real no disponible")
        df = pd.read_parquet(path)
        result = merge_embeddings(df, [
            CensusDemographicsEmbedding(),
            SpatialCoordinateEmbedding(),
        ])
        # Debe tener las nuevas columnas
        assert "census_pct_envejecimiento" in result.columns
        assert "spatial_dist_madrid_km" in result.columns
        # No debe perder filas
        assert len(result) == len(df)
        # No debe tener NaN en las columnas de embedding
        for col in result.columns:
            if col.startswith("census_") or col.startswith("spatial_"):
                assert result[col].notna().all(), f"{col} tiene NaN tras merge"

    def test_merge_missing_provincia_raises(self):
        df = pd.DataFrame({"t2m_c": [1.0]})
        with pytest.raises(ValueError, match="provincia"):
            merge_embeddings(df, [CensusDemographicsEmbedding()])

    def test_embedding_dimensions_consistent(self):
        """Ambos providers generan vectores de longitud fija."""
        census = CensusDemographicsEmbedding()
        spatial = SpatialCoordinateEmbedding()
        ec = census.get_embeddings()
        es = spatial.get_embeddings()
        assert ec.shape[1] == census.n_features + 1
        assert es.shape[1] == spatial.n_features + 1
        # Todos los vectores del mismo provider tienen la misma longitud
        assert ec.shape[0] == es.shape[0]  # mismo número de provincias
