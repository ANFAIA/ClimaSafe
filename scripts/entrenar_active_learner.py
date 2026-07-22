"""
Entrena el ActiveLearner con factores existentes + negativos sintéticos.

Útil tras limpiar scout_entrenamiento o para resetear el clasificador.
Uso: .venv/bin/python3 scripts/entrenar_active_learner.py
"""

import warnings
warnings.filterwarnings("ignore")

import sqlite3
from climasafeai.ml.active_learner import ActiveLearner

DB_PATH = "data/climasafe.db"

al = ActiveLearner()
al.ensure_table()

# Limpiar datos previos para entrenamiento fresco
conn = sqlite3.connect(DB_PATH)
conn.execute("DELETE FROM scout_entrenamiento")
conn.commit()
conn.close()

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row

rows = conn.execute("""
    SELECT nombre, tipo, categoria, clave, poblacion
    FROM factores_riesgo WHERE implementado = 1
""").fetchall()
conn.close()

for f in [dict(r) for r in rows]:
    nombre = f.get("nombre", f["clave"])
    pob = f.get("poblacion") or "general population"
    title = f"{nombre} as risk factor in heat and cold exposure"
    abstract = (
        f"Study examining {nombre} as a risk factor for heat/cold related morbidity. "
        f"Population: {pob}. Category: {f['categoria']}, type: {f['tipo']}."
    )
    al.store(title, abstract, "aceptable", fuente="sintetico")

negatives = [
    ("Cooking with olive oil", "Health benefits of olive oil in Mediterranean cuisine"),
    ("Machine learning for image recognition", "Deep convolutional networks for object detection"),
    ("Stock market prediction using LSTMs", "Financial time series forecasting with RNNs"),
    ("Quantum computing advances", "Superconducting qubits and quantum error correction"),
    ("Basketball player performance analysis", "NBA player efficiency ratings analysis"),
    ("Fashion trends in sustainable clothing", "Consumer preferences for eco-friendly fabrics"),
    ("Astrophysics of black hole mergers", "Gravitational wave observations from black hole coalescence"),
    ("Urban planning and traffic optimization", "Reinforcement learning for adaptive traffic lights"),
    ("Advances in battery technology", "Lithium-ion battery degradation and solid-state electrolytes"),
    ("Marine biology of coral reefs", "Coral bleaching events and ecosystem responses"),
    ("Shakespearean literature analysis", "Thematic analysis of tragedy in Elizabethan drama"),
    ("Robotics in manufacturing", "Automated assembly line optimization with cobots"),
    ("Cryptocurrency regulation", "Comparative analysis of digital asset regulations"),
    ("Music genre classification", "Audio feature extraction and SVM classification"),
    ("French pastry techniques", "Temperature effects on croissant lamination"),
]

for title, abstract in negatives:
    al.store(title, abstract, "irrelevante", fuente="sintetico")

r = al.retrain()
print(f"Entrenado: fitted={r['fitted']}, samples={r['samples']}")
print(f"  Positivos (aceptable): {r['aceptables']}")
print(f"  Negativos (irrelevante): {r['irrelevantes']}")

tests = [
    ("diabetes and heat risk", "risk factors for diabetes patients in extreme heat", "aceptable"),
    ("cardiovascular disease cold weather", "CVD exacerbation during winter months", "aceptable"),
    ("cooking recipes", "best pasta recipes", "irrelevante"),
    ("stock market prediction", "LSTM for stock price forecasting", "irrelevante"),
]

print("\nValidación:")
for title, abstract, expected in tests:
    v, c, _ = al.predict(title, abstract)
    ok = "✓" if v == expected else "✗"
    print(f"  {ok} {title:45s} → {str(v):12s} (conf={c:.3f})")
