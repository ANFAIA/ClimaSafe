
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from climasafeai.utils.paths import FIGURES_DIR

plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (12, 7)


def plot_distributions(df: pd.DataFrame, target_col: str = None) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    if not num_cols:
        return

    fig, axes = plt.subplots(len(num_cols), 2, figsize=(14, 4 * len(num_cols)))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        if target_col and target_col in df.columns:
            for label, grp in df.groupby(target_col):
                axes[i][0].hist(grp[col].dropna(), bins=30, alpha=0.6, label=str(label))
            axes[i][0].legend(title=target_col)
            sns.boxplot(x=target_col, y=col, data=df, ax=axes[i][1])
        else:
            axes[i][0].hist(df[col].dropna(), bins=30, color="steelblue", alpha=0.7)
            axes[i][1].boxplot(df[col].dropna(), vert=False)
            axes[i][1].set_yticklabels([col])
        axes[i][0].set_title(f"Distribución — {col}")
        axes[i][1].set_title(f"Boxplot — {col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=150)
    plt.close(fig)
    print("    distributions.png guardado")


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()
    fig, ax = plt.subplots(figsize=(max(8, len(corr) * 0.8), max(6, len(corr) * 0.7)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, ax=ax)
    ax.set_title("Matriz de correlación")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("    correlation_matrix.png guardado")


def plot_class_balance(df: pd.DataFrame, target_col: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    counts = df[target_col].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title(f"Conteo por clase — {target_col}")
    axes[0].set_ylabel("Muestras")
    axes[0].tick_params(axis="x", rotation=0)
    axes[1].pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
    axes[1].set_title(f"Proporción de clases — {target_col}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=150)
    plt.close(fig)
    print("    class_balance.png guardado")


def plot_categorical_vs_target(df: pd.DataFrame, target_col: str, max_cols: int = 6) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c != target_col][:max_cols]
    if not cat_cols:
        return
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 6))
    if len(cat_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cat_cols):
        sns.countplot(data=df, x=col, hue=target_col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Variables categóricas vs {target_col}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "categorical_vs_target.png", dpi=150)
    plt.close(fig)
    print("    categorical_vs_target.png guardado")


def plot_feature_importance(models: dict, feature_names: list) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    supported = {}
    for name, model in models.items():
        est = list(model.named_steps.values())[-1] if hasattr(model, "named_steps") else model
        if hasattr(est, "feature_importances_"):
            supported[name] = (est.feature_importances_, "Importancia (Gini)")
        elif hasattr(est, "coef_"):
            coef = np.abs(est.coef_)
            supported[name] = (coef[0] if coef.ndim > 1 else coef, "Magnitud coeficiente")
    if not supported:
        print("    Ningún modelo soporta importancia de variables")
        return

    fig, axes = plt.subplots(1, len(supported), figsize=(8 * len(supported), 7))
    if len(supported) == 1:
        axes = [axes]
    for ax, (name, (importances, xlabel)) in zip(axes, supported.items()):
        # Ajustar feature_names a la longitud real (puede diferir si hay PCA)
        n_imp = len(importances)
        names = feature_names[:n_imp] if len(feature_names) >= n_imp \
                else [f"feature_{i}" for i in range(n_imp)]
        df_imp = pd.DataFrame({"Feature": names, "Importance": importances}) \
                   .sort_values("Importance", ascending=False)
        sns.barplot(x="Importance", y="Feature", data=df_imp, ax=ax, orient="h")
        ax.set_title(f"Importancia — {name}")
        ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)
    print("    feature_importance.png guardado")


def plot_pca_variance(pca_or_X, n_components: int = None) -> None:
    """
    Curva de varianza explicada acumulada por PCA.

    Acepta dos formas de llamada:
      plot_pca_variance(pca)          # objeto PCA ya ajustado (desde artifacts/)
      plot_pca_variance(X_scaled)     # array: ajusta PCA internamente

    Muestra:
      - Varianza explicada por cada componente (barras)
      - Varianza acumulada (línea)
      - Marca el punto donde se alcanza el 95% y el 99%

    ¿Cuántas componentes usar?
      Regla práctica: conservar las que acumulen ≥ 95% de varianza.
      Puedes ser más agresivo (90%) si quieres mayor compresión.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from sklearn.decomposition import PCA as _PCA

    if hasattr(pca_or_X, "explained_variance_ratio_"):
        # Ya es un objeto PCA ajustado
        evr    = pca_or_X.explained_variance_ratio_
        cumvar = np.cumsum(evr)
    else:
        # Es un array: ajustar PCA
        pca    = _PCA(n_components=n_components)
        pca.fit(pca_or_X)
        evr    = pca.explained_variance_ratio_
        cumvar = np.cumsum(evr)

    n = len(evr)
    x = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Barras de varianza por componente
    ax1.bar(x, evr * 100, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_title("Varianza por componente")
    ax1.set_xticks(x[::max(1, n // 15)])

    # Curva acumulada
    ax2.plot(x, cumvar * 100, "b-o", lw=2, markersize=4)
    ax2.axhline(95, color="red",    linestyle="--", lw=1.5, label="95%")
    ax2.axhline(99, color="orange", linestyle="--", lw=1.5, label="99%")

    idx_95 = int(np.argmax(cumvar >= 0.95))
    idx_99 = int(np.argmax(cumvar >= 0.99))
    ax2.axvline(idx_95 + 1, color="red",    lw=1, linestyle=":")
    ax2.axvline(idx_99 + 1, color="orange", lw=1, linestyle=":")
    ax2.annotate(f"d={idx_95+1}", xy=(idx_95+1, cumvar[idx_95]*100),
                 xytext=(idx_95+2, cumvar[idx_95]*100 - 5), fontsize=9, color="red")
    ax2.set_xlabel("Número de componentes")
    ax2.set_ylabel("Varianza acumulada (%)")
    ax2.set_title("Varianza explicada acumulada")
    ax2.legend()
    ax2.grid(True)

    fig.suptitle(f"Análisis PCA ({n} componentes)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_variance.png", dpi=150)
    plt.close(fig)
    print(f"    pca_variance.png guardado  (95% varianza con d={idx_95+1} componentes)")


def plot_pairplot(df: pd.DataFrame, target_col: str, max_features: int = 6) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cols_to_plot = num_cols[:max_features] + [target_col]
    g = sns.pairplot(df[cols_to_plot], hue=target_col, diag_kind="kde",
                     plot_kws={"alpha": 0.5})
    g.figure.suptitle("Pairplot — separabilidad por clase", y=1.02, fontsize=13)
    g.figure.savefig(FIGURES_DIR / "pairplot.png", dpi=120, bbox_inches="tight")
    plt.close(g.figure)
    print("    pairplot.png guardado")


