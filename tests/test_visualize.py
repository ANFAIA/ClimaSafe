"""
test_visualize.py — Tests para climasafeai/visualization/visualize.py
"""
import numpy as np
import pandas as pd
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Helpers comunes
# ─────────────────────────────────────────────────────────────────────────────

def _numeric_df(n=80, cols=4):
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randn(n, cols),
        columns=[f"feat_{i}" for i in range(cols)],
    )


def _df_with_target(n=80):
    df = _numeric_df(n)
    df["target"] = (df["feat_0"] + df["feat_1"] > 0).astype(int)
    return df



from climasafeai.visualization.visualize import (
    plot_distributions,
    plot_correlation_matrix,
    plot_class_balance,
    plot_categorical_vs_target,
    plot_feature_importance,
    plot_pca_variance,
    plot_pairplot,
)


def test_plot_distributions_saves_png(patch_paths):
    plot_distributions(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "distributions.png").exists()


def test_plot_correlation_matrix_saves_png(patch_paths):
    plot_correlation_matrix(_numeric_df())
    assert (patch_paths["FIGURES_DIR"] / "correlation_matrix.png").exists()


def test_plot_class_balance_saves_png(patch_paths):
    plot_class_balance(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "class_balance.png").exists()


def test_plot_categorical_vs_target_with_cats(patch_paths):
    df = _df_with_target()
    df["cat_col"] = np.where(df["feat_0"] > 0, "A", "B")
    plot_categorical_vs_target(df, target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "categorical_vs_target.png").exists()


def test_plot_feature_importance_rf(patch_paths):
    """plot_feature_importance con RandomForest debe guardar el gráfico."""
    from sklearn.ensemble import RandomForestClassifier
    df = _df_with_target()
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    plot_feature_importance({"RF": rf}, feature_names=["feat_0","feat_1","feat_2","feat_3"])
    assert (patch_paths["FIGURES_DIR"] / "feature_importance.png").exists()


def test_plot_pca_variance_from_array(patch_paths):
    X = np.random.randn(100, 6)
    plot_pca_variance(X)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_pca_variance_from_pca_object(patch_paths):
    from sklearn.decomposition import PCA
    X = np.random.randn(100, 6)
    pca = PCA(n_components=5).fit(X)
    plot_pca_variance(pca)
    assert (patch_paths["FIGURES_DIR"] / "pca_variance.png").exists()


def test_plot_pairplot_saves_png(patch_paths):
    plot_pairplot(_df_with_target(), target_col="target")
    assert (patch_paths["FIGURES_DIR"] / "pairplot.png").exists()









