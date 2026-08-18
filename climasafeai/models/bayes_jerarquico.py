"""
bayes_jerarquico.py — Modelo bayesiano jerárquico de riesgo por provincia
(partial pooling). Feature BAYES-001.

Qué es: una regresión logística ORDINAL acumulativa de 3 clases
(0=SEGURO, 1=PRECAUCIÓN, 2=PELIGRO) donde cada provincia tiene su propio
efecto aleatorio ``u_j ~ Normal(0, τ)``. El partial pooling hace que las
provincias con pocos episodios se apoyen en la distribución nacional (τ
encoge sus estimaciones hacia la media) en vez de sobreajustar — el problema
exacto que BAYES-001 quiere atacar: hoy las provincias pequeñas comparten los
mismos parámetros que Madrid o Sevilla.

    score_i   = X_i·β + u_{prov[i]}            (u_j ~ N(0, τ), τ ~ HalfNormal)
    P(y_i≤k)  = σ(c_k − score_i)   k = 1, 2    (c_1 < c_2, cortes ordinales)
    β_m       ~ Normal(0, 2)                    (prior débil regularizador)

Sampler: Metropolis-Hastings PROPIO con numpy/scipy (decisión documentada en
documentacion/modelos/bayes_jerarquico.md). NO se instala pymc: para este
modelo (~50 parámetros) numpy/scipy bastan, evita arrastrar pytensor como
dependencia, y el muestreo por bloques con escalas adaptativas converge en
segundos incluso sobre el dataset completo. Consecuencia honesta: las
"divergencias" de NUTS no existen en MH — la métrica equivalente de salud de
transiciones es la tasa de aceptación por bloque (reportada); las de
convergencia global son r_hat (Gelman-Rubin) y ESS (Geyer), también
reportadas. Ver `resumen_muestreo()`.

Datos de provincia: los CSVs ``X_train_*.csv`` NO llevan provincia — el drop
es deliberado (``COLS_TO_DROP`` en build_features.py, para que el modelo no
aprenda "Madrid → riesgo alto"). La única fuente que conserva provincia por
fila son los parquets ``data/processed/dataset_{calor,frio}_labeled.parquet``
(172395×39, fecha+provincia+clase_riesgo). ``cargar_datos_entrenamiento()``
los usa como entrada; el método queda documentado en la doc de modelos.

El target ``clase_riesgo_{calor,frio}`` ya se construyó POR PROVINCIA
(labels.py ``asignar_clase_riesgo_*`` con ``por_provincia=True``): la etiqueta
es el riesgo relativo de cada provincia, no un umbral global. El modelo
jerárquico modela ese riesgo relativo con un efecto por provincia, coherente
con cómo se etiquetó.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd
from scipy.special import expit

from climasafeai.utils.paths import PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Features por defecto: variables meteorológicas que el pipeline ya usa
# (weather_indices), sin mortalidad (fuga directa de la etiqueta), sin
# identificadores (provincia/fecha). Elegidas por clase según cuál domina la
# señal (HI en calor, wind chill en frío) y SIN colinearidad fuerte entre sí
# (p.ej. t2m_c se omite: es casi función de heat_index_c/wind_chill_c y
# degrada la mezcla del sampler). Mismo criterio que el resto del pipeline: el
# modelo aprende de condiciones meteorológicas, no de "dónde".
FEATURES_POR_DEFECTO: dict[str, list[str]] = {
    "calor": ["heat_index_c", "heat_index_c_roll7", "heat_index_std", "rh"],
    "frio": ["wind_chill_c", "wind_chill_mean_roll7", "wind_chill_std", "rh"],
}

# Baseline tabular de la comparativa: los componentes del ensemble actual que
# sí se pueden reentrenar sobre una partición temporal (los joblib de models/
# vieron TODO el histórico y no sirven para validar sin fuga temporal — ver
# comparar_con_ensemble).
MODELO_BASELINE_POR_CLASE: dict[str, str] = {
    "calor": "XGBoost",
    "frio": "RandomForest",
}

CLASES = ["SEGURO", "PRECAUCION", "PELIGRO"]

# ---------------------------------------------------------------------------
# Carga de datos y partición temporal
# ---------------------------------------------------------------------------


def cargar_datos_entrenamiento(clase: str, path_parquet=None) -> pd.DataFrame:
    """Carga el dataset labelizado con provincia+fecha+clase_riesgo.

    Los CSVs ``X_train_*.csv`` / ``X_test_*.csv`` no sirven para un modelo por
    provincia: ``COLS_TO_DROP`` eliminó provincia/fecha/datetime como features
    (sesgo geográfico y temporal, ver build_features.py:45). La provincia se
    reconstruye desde los parquets labelizados, la única fuente que la
    conserva por fila (172395×39). La columna de clase ya es relativa a cada
    provincia (labels.py por_provincia=True).
    """
    if clase not in FEATURES_POR_DEFECTO:
        raise ValueError(f"clase debe ser 'calor' o 'frio', no {clase!r}")
    if path_parquet is None:
        path_parquet = PROCESSED_DATA_DIR / f"dataset_{clase}_labeled.parquet"
    df = pd.read_parquet(path_parquet)
    col_y = f"clase_riesgo_{clase}"
    if col_y not in df.columns:
        raise ValueError(
            f"{path_parquet} no tiene la columna '{col_y}' (¿parquet labelizado?)."
        )
    if "provincia" not in df.columns:
        raise ValueError(
            f"{path_parquet} no tiene la columna 'provincia'. "
            "El modelo jerárquico necesita provincia por fila."
        )
    return df


def particion_temporal_por_fecha(
    df: pd.DataFrame, test_size: float = 0.2, col_fecha: str = "fecha"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train/test por fechas distintas (mismo criterio que
    ``preprocess_data(split_by_date=True)`` de build_features.py): el test es
    el último ``test_size`` de fechas distintas del histórico. Así el modelo
    jerárquico y el baseline se comparan sobre la MISMA partición temporal y
    no hay días de la misma ola repartidos entre train y test.
    """
    df = df.copy()
    fechas = pd.to_datetime(df[col_fecha])
    fechas_unicas = np.sort(fechas.unique())
    n_test_fechas = max(1, round(len(fechas_unicas) * test_size))
    fechas_test = set(fechas_unicas[-n_test_fechas:])
    mask_test = fechas.isin(fechas_test)
    df_train, df_test = df[~mask_test], df[mask_test]
    if len(df_train) == 0 or len(df_test) == 0:
        raise ValueError(
            f"particion_temporal_por_fecha: partición vacía "
            f"(train={len(df_train)}, test={len(df_test)}). "
            f"Revisa test_size={test_size} y el rango de fechas."
        )
    return df_train, df_test


# ---------------------------------------------------------------------------
# Diagnósticos de muestreo
# ---------------------------------------------------------------------------


def r_hat(trazas_cadenas: np.ndarray) -> float:
    """Gelman-Rubin por parámetro. Entrada: (C, T) — C cadenas, T muestras.

    r_hat ≈ 1 indica convergencia (varianza entre cadenas ≈ varianza dentro).
    Regla habitual: < 1.1 aceptable.
    """
    C, T = trazas_cadenas.shape
    if C < 2:
        return np.nan
    medias = trazas_cadenas.mean(axis=1)
    var_dentro = trazas_cadenas.var(axis=1, ddof=1)
    B = T * np.var(medias, ddof=1)
    W = np.mean(var_dentro)
    if W <= 0:
        return 1.0
    var_posterior = ((T - 1) / T) * W + (1 / T) * B
    return float(np.sqrt(var_posterior / W))


def ess(traza: np.ndarray) -> float:
    """Tamaño de muestra efectivo (Geyer, secuencia positiva inicial).

    ESS = T / (1 + 2 Σ ρ̂_k), sumando autocorrelaciones positivas hasta la
    primera no positiva. Un ESS bajo indica que las muestras están muy
    autocorrelacionadas (poco informativas).
    """
    traza = np.asarray(traza, dtype=float)
    n = len(traza)
    if n < 3:
        return float(n)
    t = traza - traza.mean()
    var = float(np.dot(t, t) / n)
    if var <= 0:
        return float(n)
    rho_sum = 0.0
    for k in range(1, n // 2):
        rho = float(np.dot(t[k:], t[:-k]) / (var * (n - k)))
        if rho <= 0:
            break
        rho_sum += rho
    return float(n / (1 + 2 * rho_sum))


def intervalo_credibilidad(traza: np.ndarray, alpha: float = 0.1) -> tuple[float, float]:
    """Intervalo de credibilidad por percentiles simétricos (100·(1−α)%).

    Percentiles 5/95 en vez de HPD: más simples, invariantes a
    reparametrización y suficientes para la decisión SEGURO/PRECAUCIÓN/PELIGRO.
    """
    lo = np.percentile(traza, 100 * alpha / 2)
    hi = np.percentile(traza, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

_OBJETIVO_ACEPTACION = 0.234  # óptimo teórico de MH para dimensiones altas


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(np.asarray(z, dtype=float))


def _log_prior(beta, c1, c2, u, tau) -> float:
    """Log prior: β~N(0,2), c1~N(0,2), δ=log(c2−c1)~N(0,1), u_j~N(0,τ), τ~HalfNorm(0,1)."""
    lp = float(np.sum(-0.5 * (beta / 2.0) ** 2))
    lp += float(-0.5 * c1**2)
    delta = np.log(max(c2 - c1, 1e-6))
    lp += float(-0.5 * delta**2)
    lp += float(np.sum(-0.5 * (u / max(tau, 1e-6)) ** 2 - np.log(max(tau, 1e-6))))
    lp += float(-0.5 * max(tau, 1e-6) ** 2)
    return lp


def _log_likelihood(X, y, prov_idx, beta, c1, c2, u) -> float:
    """Log-verosimilitud del logit ordinal acumulativo, vectorizada."""
    score = X @ beta + u[prov_idx]
    p_le1 = np.clip(_sigmoid(c1 - score), 1e-12, 1 - 1e-12)
    p_le2 = np.clip(_sigmoid(c2 - score), 1e-12, 1 - 1e-12)
    p0 = p_le1
    p1 = np.clip(p_le2 - p_le1, 1e-12, 1 - 1e-12)
    p2 = np.clip(1.0 - p_le2, 1e-12, 1 - 1e-12)
    probs = np.stack([p0, p1, p2], axis=1)
    return float(np.sum(np.log(probs[np.arange(len(y)), y])))


def _adaptar_escala(sigma: float, tasa: float, t: int) -> float:
    """Adaptación de escala de propuesta durante warmup (Roberts & Rosenthal).

    ``tasa`` debe ser la aceptación de la ventana RECIENTE (no la acumulada),
    si no la adaptación se amortigua y no alcanza la escala correcta.
    """
    gamma = 0.1 if t < 500 else 0.05
    return sigma * np.exp(gamma * (tasa - _OBJETIVO_ACEPTACION))


def _inicializacion_frecuentista(
    X: np.ndarray, y: np.ndarray, prov_idx: np.ndarray, J: int, random_state: int = 42
) -> tuple:
    """Punto de partida de las cadenas cerca del modo de la posterior.

    MH arrancando en 0 (β=0, c=−0.5, u=0) necesita mucho burn-in para llegar
    al modo (c1≈7, u≈±4 en estos datos) y las cadenas convergen mal. Un ajuste
    frecuentista rápido — LogisticRegression multinomial sobre features +
    dummies de provincia — da β, u, c1/c2 del orden correcto; el warmup solo
    tiene que refinar y las cadenas convergen con r_hat razonable.
    """
    from sklearn.linear_model import LogisticRegression

    Z = np.hstack([X, np.eye(J)[prov_idx]])
    clf = LogisticRegression(
        solver="lbfgs", C=10.0, max_iter=2000, random_state=random_state
    )
    clf.fit(Z, y)
    beta = clf.coef_[:, : X.shape[1]].mean(axis=0)
    u = clf.coef_[:, X.shape[1]:].mean(axis=0)
    c1 = float(np.mean(clf.intercept_[:2]))
    c2 = float(np.mean(clf.intercept_[1:]) + 1.0)
    if c2 <= c1:
        c2 = c1 + 1.0
    tau = float(np.std(u)) if np.std(u) > 0.1 else 1.0
    return beta, c1, c2, u, tau


def _sample_chain(
    X: np.ndarray,
    y: np.ndarray,
    prov_idx: np.ndarray,
    rng: np.random.Generator,
    n_warmup: int,
    n_muestras: int,
    init: tuple | None = None,
) -> dict:
    """Metropolis-Hastings por bloques para una cadena.

    Bloques: β (por coordenada), c1, δ=log(c2−c1), u (vector), log τ. Cada
    bloque se propone con Normal centrada en el valor actual y escala
    adaptativa (objetivo 0.234) durante warmup. ``init`` (opcional) es la
    inicialización frecuentista: las cadenas arrancan cerca del modo y el
    warmup solo refina. Devuelve las trazas post-warmup y la tasa de
    aceptación por bloque.
    """
    F = X.shape[1]
    J = len(np.unique(prov_idx))
    if init is not None:
        beta, c1, c2, u, tau = init
        beta = np.asarray(beta, dtype=float)
        u = np.asarray(u, dtype=float)
        delta = np.log(max(c2 - c1, 1e-6))
        log_tau = np.log(max(tau, 1e-6))
    else:
        beta = np.zeros(F)
        c1 = -0.5
        delta = np.log(1.0)  # c2 = c1 + exp(delta) = -0.5 + 1 = 0.5
        u = np.zeros(J)
        log_tau = 0.0  # τ = 1

    def _log_post(beta_, c1_, delta_, u_, log_tau_):
        tau_ = np.exp(log_tau_)
        c2_ = c1_ + np.exp(delta_)
        return _log_prior(beta_, c1_, c2_, u_, tau_) + _log_likelihood(
            X, y, prov_idx, beta_, c1_, c2_, u_
        )

    esc = {"beta": 0.1, "c1": 0.1, "delta": 0.1, "u": 0.2, "tau": 0.1}
    lp = _log_post(beta, c1, delta, u, log_tau)

    n_total = n_warmup + n_muestras
    trazas = {
        "beta": np.empty((n_muestras, F)),
        "c1": np.empty(n_muestras),
        "c2": np.empty(n_muestras),
        "u": np.empty((n_muestras, J)),
        "tau": np.empty(n_muestras),
    }
    aceptados = {bloque: 0 for bloque in esc}
    # Ventana reciente de aceptación por bloque para la adaptación de escala:
    # con la tasa acumulada la adaptación converge demasiado lento (r_hat alto).
    ventana = {bloque: deque(maxlen=100) for bloque in esc}

    def _aceptar(lp_prop, lp_actual, bloque):
        """Decisión MH + registro de aceptación (acumulada y ventana)."""
        if np.log(rng.uniform()) < lp_prop - lp_actual:
            ventana[bloque].append(1)
            return True
        ventana[bloque].append(0)
        return False

    for t in range(n_total):
        # ── β, POR COORDENADA ──
        # Un bloque vectorial mezcla mal: las coordenadas están correlacionadas
        # (la posterior es una cresta estrecha) y la propuesta conjunta casi
        # nunca se acepta. Proponer cada β_j por separado da aceptación alta y
        # cadenas que recorren la cresta (r_hat razonable).
        for j in range(F):
            prop = beta.copy()
            prop[j] += rng.normal(0, esc["beta"])
            lp_prop = _log_post(prop, c1, delta, u, log_tau)
            if _aceptar(lp_prop, lp, "beta"):
                beta, lp = prop, lp_prop
                aceptados["beta"] += 1
        # ── c1 ──
        prop = c1 + rng.normal(0, esc["c1"])
        lp_prop = _log_post(beta, prop, delta, u, log_tau)
        if _aceptar(lp_prop, lp, "c1"):
            c1, lp = prop, lp_prop
            aceptados["c1"] += 1
        # ── δ (c2 > c1) ──
        prop = delta + rng.normal(0, esc["delta"])
        lp_prop = _log_post(beta, c1, prop, u, log_tau)
        if _aceptar(lp_prop, lp, "delta"):
            delta, lp = prop, lp_prop
            aceptados["delta"] += 1
        # ── u (efectos por provincia) ──
        prop = u + rng.normal(0, esc["u"], size=J)
        lp_prop = _log_post(beta, c1, delta, prop, log_tau)
        if _aceptar(lp_prop, lp, "u"):
            u, lp = prop, lp_prop
            aceptados["u"] += 1
        # ── log τ ──
        prop = log_tau + rng.normal(0, esc["tau"])
        lp_prop = _log_post(beta, c1, delta, u, prop)
        if _aceptar(lp_prop, lp, "tau"):
            log_tau, lp = prop, lp_prop
            aceptados["tau"] += 1

        if t < n_warmup:
            for bloque, sigma in esc.items():
                tasa_ventana = np.mean(ventana[bloque]) if ventana[bloque] else 0.5
                esc[bloque] = _adaptar_escala(sigma, tasa_ventana, t)
        else:
            i = t - n_warmup
            trazas["beta"][i] = beta
            trazas["c1"][i] = c1
            trazas["c2"][i] = c1 + np.exp(delta)
            trazas["u"][i] = u
            trazas["tau"][i] = np.exp(log_tau)

    trazas["aceptacion"] = {k: v / n_total for k, v in aceptados.items()}
    return trazas


class ModeloJerarquico:
    """Regresión logística ordinal jerárquica por provincia (partial pooling).

    Uso típico:
        df = cargar_datos_entrenamiento("calor")
        modelo = ModeloJerarquico(clase="calor")
        modelo.fit(df)
        resumen = modelo.resumen_muestreo()      # r_hat, ESS, aceptación
        pred = modelo.predecir(df_test)          # clase + intervalo de credibilidad
    """

    def __init__(
        self,
        clase: str = "calor",
        features: list[str] | None = None,
        random_state: int = 42,
    ):
        if clase not in FEATURES_POR_DEFECTO:
            raise ValueError(f"clase debe ser 'calor' o 'frio', no {clase!r}")
        self.clase = clase
        self.features = list(features) if features else list(FEATURES_POR_DEFECTO[clase])
        self.random_state = random_state
        self._col_y = f"clase_riesgo_{clase}"
        self._provincias: list[str] = []
        self._prov_idx: dict[str, int] = {}
        self._medias: pd.Series = pd.Series(dtype=float)
        self._scaler = None
        self._cadenas: list[dict] = []

    # -----------------------------------------------------------------------
    # Preparación de features (mismas para jerárquico y baseline)
    # -----------------------------------------------------------------------

    def _preparar_X(self, df: pd.DataFrame, ajustar: bool = False) -> np.ndarray:
        """Features numéricas, nulos rellenados y escaladas.

        Los nulos se rellenan con la media de TRAIN (guardada en ``_medias``)
        y el escalado se ajusta solo en train (``ajustar=True``) y se aplica
        después — el test no influye en ningún ajuste, como el resto del
        pipeline (temporal_cv._prepare_fold).
        """
        faltan = [c for c in self.features if c not in df.columns]
        if faltan:
            raise KeyError(f"Faltan features: {faltan}")
        X = df[self.features].astype(float)
        if ajustar:
            self._medias = X.mean()
        X = X.fillna(self._medias)
        if self._scaler is None:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
        if ajustar:
            X = self._scaler.fit_transform(X)
        else:
            X = self._scaler.transform(X)
        return np.asarray(X)

    def _prov_idx_array(self, provincias: pd.Series) -> np.ndarray:
        """Índices de provincia; las desconocidas van a un grupo nuevo con u=0
        (media del prior: el partial pooling las trata como la provincia media)."""
        return np.array(
            [self._prov_idx.get(p, len(self._provincias)) for p in provincias]
        )

    # -----------------------------------------------------------------------
    # Entrenamiento
    # -----------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        col_y: str | None = None,
        col_provincia: str = "provincia",
        n_cadenas: int = 2,
        n_warmup: int = 400,
        n_muestras: int = 200,
    ):
        """Entrena con Metropolis-Hastings (n_cadenas cadenas independientes).

        Parámetros
        ----------
        df : DataFrame con las features, ``col_provincia`` y ``col_y``.
        col_y : columna de la clase 0/1/2. Por defecto ``clase_riesgo_{clase}``
            (la del parquet labelizado).
        n_cadenas : cadenas paralelas para r_hat/ESS (2 por defecto).
        n_warmup, n_muestras : iteraciones de calentamiento y guardadas.
        """
        if col_y is not None:
            self._col_y = col_y
        if self._col_y not in df.columns:
            raise KeyError(f"Falta la columna target '{self._col_y}' en el DataFrame.")
        if col_provincia not in df.columns:
            raise KeyError(f"Falta la columna '{col_provincia}' en el DataFrame.")

        self._provincias = sorted(df[col_provincia].dropna().unique())
        self._prov_idx = {p: i for i, p in enumerate(self._provincias)}

        X = self._preparar_X(df, ajustar=True)
        y = df[self._col_y].to_numpy(dtype=int)
        prov_idx = self._prov_idx_array(df[col_provincia])
        J = len(self._provincias)

        # Inicialización frecuentista: las cadenas arrancan cerca del modo en
        # vez de en 0 (ver _inicializacion_frecuentista).
        init = _inicializacion_frecuentista(X, y, prov_idx, J, random_state=self.random_state)

        self._cadenas = []
        for cadena in range(n_cadenas):
            rng = np.random.default_rng(self.random_state + cadena)
            self._cadenas.append(
                _sample_chain(X, y, prov_idx, rng, n_warmup, n_muestras, init=init)
            )
        return self

    # -----------------------------------------------------------------------
    # Salida del muestreo
    # -----------------------------------------------------------------------

    def _concatenar(self, param: str) -> np.ndarray:
        """Concatena la traza de un parámetro de todas las cadenas: (M,) o (M,F)."""
        return np.concatenate([c[param] for c in self._cadenas], axis=0)

    def resumen_muestreo(self) -> pd.DataFrame:
        """Tabla de diagnóstico del muestreo por parámetro.

        Columnas: mediana, r_hat (Gelman-Rubin entre cadenas), ESS (Geyer,
        suma sobre cadenas) y la tasa de aceptación del bloque. La columna
        ``divergencias`` es siempre 0 con la nota de que MH no produce las
        transiciones divergentes de NUTS/HMC: su equivalente, la tasa de
        aceptación por bloque, sí se reporta (documentacion/modelos/).
        """
        if not self._cadenas:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")
        filas = []
        J = len(self._provincias)
        for i in range(self.features.__len__()):
            filas.append((f"beta_{self.features[i]}", "beta"))
        filas.append(("c1", "c1"))
        filas.append(("c2", "c2"))
        for j in range(J):
            filas.append((f"u_{self._provincias[j]}", "u"))
        filas.append(("tau", "tau"))

        resultados = []
        for nombre, bloque in filas:
            if bloque == "beta":
                arr = self._concatenar("beta")[:, self.features.index(nombre.split("_", 1)[1])]
            elif bloque == "u":
                j = self._provincias.index(nombre.split("_", 1)[1])
                arr = self._concatenar("u")[:, j]
            else:
                arr = self._concatenar(bloque)
            if bloque == "beta":
                idx_feat = self.features.index(nombre.split("_", 1)[1])
                por_cadena = np.stack([c["beta"][:, idx_feat] for c in self._cadenas])
            elif bloque == "u":
                j = self._provincias.index(nombre.split("_", 1)[1])
                por_cadena = np.stack([c["u"][:, j] for c in self._cadenas])
            else:
                por_cadena = np.stack([c[bloque] for c in self._cadenas])
            # El bloque del sampler para c2 se llama "delta" (propuesta en log
            # del desplazamiento c2−c1); la clave de aceptación es esa.
            bloque_aceptacion = "delta" if bloque == "c2" else bloque
            resultados.append({
                "parametro": nombre,
                "mediana": float(np.median(arr)),
                "r_hat": r_hat(por_cadena),
                "ess": float(sum(ess(c) for c in por_cadena)),
                "aceptacion": self._cadenas[0]["aceptacion"].get(
                    bloque_aceptacion, float("nan")
                ),
                "divergencias": 0,
            })
        return pd.DataFrame(resultados)

    # -----------------------------------------------------------------------
    # Predicción con intervalo de credibilidad
    # -----------------------------------------------------------------------

    def predecir(
        self,
        df: pd.DataFrame,
        col_provincia: str = "provincia",
        alpha: float = 0.1,
        max_muestras: int = 400,
    ) -> pd.DataFrame:
        """Predice clase + intervalo de credibilidad (100·(1−α)%) por fila.

        Para cada muestra posterior se computa la distribución (p0, p1, p2)
        de cada fila; la salida da la MEDIANA (predicción puntual) y los
        percentiles α/2 y 1−α/2 (intervalo de credibilidad) de la
        probabilidad de riesgo (P(1)+P(2)) y de cada clase.

        Traducción a SEGURO/PRECAUCIÓN/PELIGRO: la clase puntual sale de la
        cascada ``apply_class_thresholds`` de predict_model.py aplicada a las
        medianas (P(2)≥t2 → PELIGRO; P(1)+P(2)≥t1 → PRECAUCIÓN; si no, SEGURO)
        con los umbrales calibrados ``CLASS_THRESHOLDS_RECOMENDADOS`` — la
        misma política del ensemble. Si los intervalos de las dos clases más
        probables se solapan, ``clase_incierta`` avisa de que la mediana no
        basta para decidir.
        """
        if not self._cadenas:
            raise RuntimeError("Modelo no entrenado. Llama a fit() primero.")
        if col_provincia not in df.columns:
            raise KeyError(f"Falta la columna '{col_provincia}' en el DataFrame.")

        X = self._preparar_X(df)
        prov_idx = self._prov_idx_array(df[col_provincia])
        n = len(df)

        # Muestras posteriores conjuntas (concatenadas), con tope para que los
        # percentiles sean estables sin explotar memoria en el dataset completo.
        M_total = sum(len(c["beta"]) for c in self._cadenas)
        paso = max(1, M_total // max_muestras)
        muestras = []
        for c in self._cadenas:
            for i in range(0, len(c["beta"]), paso):
                muestras.append((c["beta"][i], c["c1"][i], c["c2"][i], c["u"][i]))
        if len(muestras) > max_muestras:  # no debería ocurrir, por seguridad
            muestras = muestras[:: len(muestras) // max_muestras]
        M = len(muestras)

        # Acumuladores por chunk de filas para no materializar (N, M, 3).
        prob_riesgo = np.empty((n, M))
        prob_clases = np.empty((n, 3, M))
        CHUNK = 2000
        for inicio in range(0, n, CHUNK):
            fin = min(inicio + CHUNK, n)
            Xc = X[inicio:fin]
            pidx = prov_idx[inicio:fin]
            for m, (beta, c1, c2, u) in enumerate(muestras):
                score = Xc @ beta + u[pidx]
                p_le1 = _sigmoid(c1 - score)
                p_le2 = _sigmoid(c2 - score)
                p0 = p_le1
                p1 = p_le2 - p_le1
                p2 = 1.0 - p_le2
                prob_riesgo[inicio:fin, m] = p1 + p2
                prob_clases[inicio:fin, 0, m] = p0
                prob_clases[inicio:fin, 1, m] = p1
                prob_clases[inicio:fin, 2, m] = p2

        q_lo, q_hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
        p_riesgo_med = np.median(prob_riesgo, axis=1)
        p_riesgo_lo = np.percentile(prob_riesgo, q_lo, axis=1)
        p_riesgo_hi = np.percentile(prob_riesgo, q_hi, axis=1)
        medianas_clase = np.median(prob_clases, axis=2)  # (n, 3)

        clases = traducir_a_clase(medianas_clase, clase=self.clase)

        # Clase incierta: si el intervalo de la clase elegida se solapa con el
        # de alguna otra clase, la mediana no decide por sí sola.
        lo = np.percentile(prob_clases, q_lo, axis=2)  # (n, 3)
        hi = np.percentile(prob_clases, q_hi, axis=2)  # (n, 3)
        lo_sel = lo[np.arange(n), clases]
        hi_sel = hi[np.arange(n), clases]
        solape = (lo <= hi_sel[:, None]) & (hi >= lo_sel[:, None])  # (n, 3)
        solape[np.arange(n), clases] = False
        incierta = solape.any(axis=1)

        return pd.DataFrame({
            "provincia": df[col_provincia].to_numpy(),
            "clase_predicha": clases,
            "clase_predicha_label": [CLASES[c] for c in clases],
            "clase_incierta": incierta,
            "prob_riesgo_mediana": p_riesgo_med.round(4),
            "prob_riesgo_lower": p_riesgo_lo.round(4),
            "prob_riesgo_upper": p_riesgo_hi.round(4),
            "prob0_mediana": medianas_clase[:, 0].round(4),
            "prob1_mediana": medianas_clase[:, 1].round(4),
            "prob2_mediana": medianas_clase[:, 2].round(4),
        })


def traducir_a_clase(probs: np.ndarray, clase: str = "calor") -> np.ndarray:
    """Traduce probabilidades (n,3) a clases 0/1/2 con la cascada del pipeline.

    Delega en ``apply_class_thresholds`` (predict_model.py): P(2) ≥ t2 →
    PELIGRO; P(1)+P(2) ≥ t1 → PRECAUCIÓN; si no → SEGURO. Los umbrales son
    los calibrados ``CLASS_THRESHOLDS_RECOMENDADOS`` de cada clase, los mismos
    que usa el ensemble actual — así el jerárquico y el ensemble hablan el
    mismo idioma de decisión.
    """
    from climasafeai.models.predict_model import (
        apply_class_thresholds,
        CLASS_THRESHOLDS_RECOMENDADOS,
    )

    u = CLASS_THRESHOLDS_RECOMENDADOS.get(clase, {"t1": 0.25, "t2": 0.10})
    return apply_class_thresholds(np.asarray(probs), t1=u["t1"], t2=u["t2"])


# ---------------------------------------------------------------------------
# Comparativa contra el ensemble actual
# ---------------------------------------------------------------------------


def _brier_multiclase(y: np.ndarray, proba: np.ndarray) -> float:
    onehot = np.zeros((len(y), 3))
    onehot[np.arange(len(y)), y] = 1
    return float(np.mean((onehot - proba) ** 2))


def _submuestra_estratificada(
    df: pd.DataFrame, n: int, col_provincia: str = "provincia", random_state: int = 42
) -> pd.DataFrame:
    """Submuestra de ``n`` filas con peso uniforme POR PROVINCIA.

    La posterior con el train completo (137k filas) es tan picuda que el MH
    converge mal (r_hat alto); con una submuestra la posterior es más suave y
    las cadenas mezclan. Los pesos 1/tamaño_provincia hacen que cada provincia
    contribuya una proporción similar, sin perder ninguna.
    """
    if len(df) <= n:
        return df
    pesos = 1.0 / df[col_provincia].map(df[col_provincia].value_counts())
    return df.sample(n=n, random_state=random_state, weights=pesos)


def comparar_con_ensemble(
    df: pd.DataFrame,
    clase: str,
    features: list[str] | None = None,
    test_size: float = 0.2,
    n_cadenas: int = 2,
    n_warmup: int = 400,
    n_muestras: int = 200,
    random_state: int = 42,
    col_y: str | None = None,
    max_filas_train: int | None = None,
) -> dict:
    """Compara el jerárquico contra el baseline tabular del ensemble actual.

    La comparación es sobre la MISMA partición temporal (último ``test_size``
    de fechas distintas) y las MISMAS features. El baseline es el componente
    tabular del ensemble (XGBoost para calor, RandomForest para frío — los
    mismos modelos y hiperparámetros de ``temporal_cv._build_models_cv``)
    REENTRENADO en esa partición: los joblib de ``models/`` ya vieron todo el
    histórico y compararlos sobre un test temporal sería fuga (el modelo ya
    habría "visto" esas fechas).

    ``max_filas_train`` (solo para la demo): submuestra estratificada por
    provincia del train temporal, para que el MH converja con r_hat razonable.
    El baseline se entrena en la misma submuestra — la comparación sigue
    siendo justa, sobre las mismas filas y la misma partición.

    Devuelve un dict con:
      - ``tabla``: DataFrame con una fila por provincia + fila TOTAL. Columnas:
        n_test, accuracy/f1/brier del baseline y del jerárquico, cobertura del
        intervalo 90% de prob_riesgo (solo el jerárquico da intervalos), y
        ganancia_f1 (jerárquico − baseline).
      - ``prediccion``: salida de ``ModeloJerarquico.predecir`` sobre test.
      - ``modelo``: el ``ModeloJerarquico`` entrenado.
    """
    if clase not in FEATURES_POR_DEFECTO:
        raise ValueError(f"clase debe ser 'calor' o 'frio', no {clase!r}")
    col_y = col_y or f"clase_riesgo_{clase}"
    df_train, df_test = particion_temporal_por_fecha(df, test_size)
    if max_filas_train is not None and len(df_train) > max_filas_train:
        df_train = _submuestra_estratificada(df_train, max_filas_train)

    modelo = ModeloJerarquico(clase=clase, features=features, random_state=random_state)
    modelo.fit(df_train, col_y=col_y, n_cadenas=n_cadenas, n_warmup=n_warmup, n_muestras=n_muestras)
    pred = modelo.predecir(df_test)
    X_train = modelo._preparar_X(df_train)
    X_test = modelo._preparar_X(df_test)
    y_test = df_test[col_y].to_numpy(dtype=int)

    # Baseline tabular del ensemble, reentrenado en la misma partición.
    from climasafeai.models.temporal_cv import _build_models_cv

    baseline = _build_models_cv(knn_k=5, clase=clase)[MODELO_BASELINE_POR_CLASE[clase]]
    baseline.fit(X_train, df_train[col_y])
    proba_base = np.asarray(baseline.predict_proba(X_test))
    clase_base = baseline.predict(X_test)
    y_bin_test = (y_test > 0).astype(int)

    from sklearn.metrics import accuracy_score, f1_score

    def _metricas(clase_pred, proba, y_masked):
        return {
            "acc": accuracy_score(y_masked, clase_pred),
            "f1": f1_score(y_masked, clase_pred, average="macro", zero_division=0),
            "brier": _brier_multiclase(y_masked, proba),
        }

    filas = []
    provincias = sorted(df_test["provincia"].unique())
    for prov in provincias + ["TOTAL"]:
        if prov == "TOTAL":
            mask = np.ones(len(df_test), dtype=bool)
        else:
            mask = (df_test["provincia"] == prov).to_numpy()
        if mask.sum() == 0:
            continue
        m_base = _metricas(clase_base[mask], proba_base[mask], y_test[mask])
        m_hier = _metricas(
            pred["clase_predicha"].to_numpy()[mask],
            medianas_clase_para(pred)[mask],
            y_test[mask],
        )
        lo = pred["prob_riesgo_lower"].to_numpy()[mask]
        hi = pred["prob_riesgo_upper"].to_numpy()[mask]
        cobertura = float(np.mean((lo <= y_bin_test[mask]) & (y_bin_test[mask] <= hi)))
        filas.append({
            "provincia": prov,
            "n_test": int(mask.sum()),
            "acc_baseline": round(m_base["acc"], 4),
            "f1_baseline": round(m_base["f1"], 4),
            "brier_baseline": round(m_base["brier"], 4),
            "acc_jerarquico": round(m_hier["acc"], 4),
            "f1_jerarquico": round(m_hier["f1"], 4),
            "brier_jerarquico": round(m_hier["brier"], 4),
            "cobertura_90": round(cobertura, 4),
            "ganancia_f1": round(m_hier["f1"] - m_base["f1"], 4),
        })

    return {
        "tabla": pd.DataFrame(filas),
        "prediccion": pred,
        "modelo": modelo,
    }


def medianas_clase_para(pred: pd.DataFrame) -> np.ndarray:
    """Matriz (n,3) de medianas por clase desde el DataFrame de ``predecir``."""
    return np.stack(
        [pred["prob0_mediana"], pred["prob1_mediana"], pred["prob2_mediana"]], axis=1
    ).astype(float)


def comparativa_muchos_pocos_datos(
    tabla: pd.DataFrame, n_episodios_por_provincia: dict[str, int]
) -> pd.DataFrame:
    """Beneficio del jerárquico según el volumen de datos de la provincia.

    Divide las provincias en dos grupos (pocos/muchos) según el número de
    EPISODIOS DE RIESGO en train — filas con clase > 0 (PRECAUCIÓN+PELIGRO),
    no filas totales: el parquet tiene una fila por (provincia, fecha) con el
    mismo rango para todas, así que el conteo de filas no discrimina. Son los
    episodios los que varían (las provincias pequeñas degradan PELIGRO por
    min_mortalidad_peligro y tienen menos avisos). Corte en la mediana.

    Es la razón de ser del modelo: en provincias con pocos episodios el
    partial pooling debe reducir la ventaja del baseline o superarlo, porque
    la estimación se apoya en la distribución nacional en vez de sobreajustar.
    """
    if "TOTAL" in tabla["provincia"].to_numpy():
        tabla = tabla[tabla["provincia"] != "TOTAL"].copy()
    tabla = tabla.copy()
    tabla["n_episodios"] = tabla["provincia"].map(n_episodios_por_provincia)
    if tabla["n_episodios"].isna().any():
        raise ValueError(
            "comparativa_muchos_pocos_datos: faltan provincias en "
            "n_episodios_por_provincia."
        )
    corte = tabla["n_episodios"].median()
    tabla["grupo"] = np.where(tabla["n_episodios"] < corte, "pocos_datos", "muchos_datos")
    grupos = []
    for grupo, sub in tabla.groupby("grupo", sort=False):
        grupos.append({
            "grupo": grupo,
            "n_provincias": len(sub),
            "episodios_medio": int(round(sub["n_episodios"].mean())),
            "f1_baseline_medio": round(sub["f1_baseline"].mean(), 4),
            "f1_jerarquico_medio": round(sub["f1_jerarquico"].mean(), 4),
            "brier_baseline_medio": round(sub["brier_baseline"].mean(), 4),
            "brier_jerarquico_medio": round(sub["brier_jerarquico"].mean(), 4),
            "cobertura_90_media": round(sub["cobertura_90"].mean(), 4),
            "ganancia_f1_media": round(sub["ganancia_f1"].mean(), 4),
        })
    return pd.DataFrame(grupos)


# ---------------------------------------------------------------------------
# Demo / evidencia (BAYES-001)
# ---------------------------------------------------------------------------


def _demo(
    clase: str,
    n_warmup: int = 1500,
    n_muestras: int = 600,
    test_size: float = 0.2,
    max_filas_train: int = 15000,
):
    """Entrenamiento con los datos completos y comparativa con el ensemble.

    Imprime la salida del muestreo (r_hat/ESS/aceptación), la tabla por
    provincia y agregado, la comparativa pocos/muchos datos, y deja la tabla
    en reports/ como evidencia reproducible.

    ``max_filas_train``: submuestra estratificada por provincia del train
    temporal. Con el train completo (137k filas) la posterior es tan picuda
    que el Metropolis-Hastings converge mal (r_hat alto); la submuestra da
    r_hat/ESS aceptables. El baseline se entrena en la misma submuestra, así
    que la comparación sigue siendo justa.
    """
    from climasafeai.utils.paths import REPORTS_DIR

    print(f"\n{'=' * 78}\n  BAYES-001 — demo del modelo jerárquico ({clase})\n{'=' * 78}")
    df = cargar_datos_entrenamiento(clase)
    print(f"Datos: {df.shape} filas | provincias: {df['provincia'].nunique()} "
          f"| fechas: {pd.to_datetime(df['fecha']).min().date()} → {pd.to_datetime(df['fecha']).max().date()}")

    df_train, df_test = particion_temporal_por_fecha(df, test_size)
    df_train_demo = df_train
    if len(df_train) > max_filas_train:
        df_train_demo = _submuestra_estratificada(df_train, max_filas_train)
        print(f"Partición temporal (test_size={test_size}): train {len(df_train)} filas "
              f"({pd.to_datetime(df_train['fecha']).max().date()} y antes) | "
              f"test {len(df_test)} filas (desde {pd.to_datetime(df_test['fecha']).min().date()})")
        print(f"Train de la demo: submuestra estratificada por provincia de "
              f"{len(df_train_demo)} filas (la posterior con {len(df_train)} filas es "
              f"tan picuda que MH no converge; el baseline usa la misma submuestra)")
    else:
        print(f"Partición temporal (test_size={test_size}): train {len(df_train)} filas "
              f"({pd.to_datetime(df_train['fecha']).max().date()} y antes) | "
              f"test {len(df_test)} filas (desde {pd.to_datetime(df_test['fecha']).min().date()})")

    print(f"\n--- Muestreo Metropolis-Hastings (2 cadenas, "
          f"warmup={n_warmup}, muestras={n_muestras}) ---")
    modelo = ModeloJerarquico(clase=clase)
    modelo.fit(df_train_demo, n_cadenas=2, n_warmup=n_warmup, n_muestras=n_muestras)
    resumen = modelo.resumen_muestreo()
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(resumen.to_string(index=False))
    print("\nNota: 'divergencias'=0 porque MH no produce transiciones divergentes "
          "(diagnóstico de NUTS/HMC); su equivalente, la tasa de aceptación por "
          "bloque, va en la columna 'aceptacion'.")

    print(f"\n--- Comparativa vs {MODELO_BASELINE_POR_CLASE[clase]} (mismas features, "
          f"misma partición temporal) ---")
    comp = comparar_con_ensemble(
        df, clase=clase, n_cadenas=2, n_warmup=n_warmup, n_muestras=n_muestras,
        test_size=test_size, max_filas_train=max_filas_train,
    )
    tabla = comp["tabla"]
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(tabla.to_string(index=False))

    n_episodios_por_provincia = (
        df_train[df_train[f"clase_riesgo_{clase}"] > 0].groupby("provincia").size().to_dict()
    )
    print("\n--- Comparativa provincias con pocos vs muchos episodios de riesgo ---")
    comparativa = comparativa_muchos_pocos_datos(tabla, n_episodios_por_provincia)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(comparativa.to_string(index=False))

    prov_menos = min(n_episodios_por_provincia, key=n_episodios_por_provincia.get)
    prov_mas = max(n_episodios_por_provincia, key=n_episodios_por_provincia.get)
    print(f"\nProvincia con MENOS episodios en train: {prov_menos} "
          f"({n_episodios_por_provincia[prov_menos]} episodios de riesgo)")
    print(f"Provincia con MÁS episodios en train: {prov_mas} "
          f"({n_episodios_por_provincia[prov_mas]} episodios de riesgo)")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTS_DIR / f"comparativa_bayes_vs_ensemble_{clase}.csv", index=False)
    comparativa.to_csv(REPORTS_DIR / f"comparativa_pocos_muchos_{clase}.csv", index=False)
    print(f"\nTablas guardadas en reports/comparativa_bayes_vs_ensemble_{clase}.csv "
          f"y reports/comparativa_pocos_muchos_{clase}.csv")
    return comp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo del modelo jerárquico BAYES-001")
    parser.add_argument("--clase", choices=["calor", "frio"], default="calor")
    parser.add_argument("--n-warmup", type=int, default=1500)
    parser.add_argument("--n-muestras", type=int, default=600)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-filas-train", type=int, default=15000)
    args = parser.parse_args()
    _demo(
        args.clase, n_warmup=args.n_warmup, n_muestras=args.n_muestras,
        test_size=args.test_size, max_filas_train=args.max_filas_train,
    )
