#!/usr/bin/env python3
"""META-001 — Mide las métricas de éxito del PRD en producción.

Fuentes de datos:
  - data/processed/{X,y}_test_{calor,frio}.csv → Recall y precisión de modelos
  - data/climasafe.db → Uso real (historial_consultas, perfiles)
  - logs/{bot,web,mcp}.log → Tráfico por canal

Métricas del PRD (documentacion/prd.md §Métricas de éxito):
  1. Recall de clases de riesgo (no perderse días peligrosos)
  2. Precisión de avisos (falsas alarmas asumibles)
  3. Anticipación (días de antelación)
  4. Cobertura (provincias × días con aviso fiable)
  5. Uso real (consultas bot/web/API, % con perfil personalizado)

Líneas base del PRD:
  - Recall_riesgo: XGBoost 0.668 (calor), RF 0.612 (frío), LSTM 0.737/0.708
  - Precisión: umbrales calibrados

Uso:
  python scripts/prd_metrics.py [--json]
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ── Líneas base del PRD ──────────────────────────────────────────────
PRD_BASELINES = {
    "recall_riesgo_calor": 0.668,
    "recall_riesgo_frio": 0.612,
    "recall_lstm_calor": 0.737,
    "recall_lstm_frio": 0.708,
}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
DB_PATH = DATA_DIR / "climasafe.db"
LOGS_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"


# ── 1. Métricas de modelo ────────────────────────────────────────────

def load_test_data(clase: str):
    """Carga X_test, y_test y el modelo desplegado para una clase."""
    X = pd.read_csv(PROCESSED_DIR / f"X_test_{clase}.csv")
    y = pd.read_csv(PROCESSED_DIR / f"y_test_{clase}.csv").values.ravel()
    model_path = MODELS_DIR / f"{'XGBoost' if clase == 'calor' else 'RandomForest'}_{clase}.joblib"
    model = joblib.load(model_path)
    return X, y, model


def compute_model_metrics(clase: str) -> dict:
    """Calcula recall, precisión, F1 y Recall_riesgo para un modelo desplegado."""
    X, y, model = load_test_data(clase)
    y_pred = model.predict(X)

    # Recall_riesgo = recall medio de clases 1 y 2
    recall_riesgo = recall_score(y, y_pred, labels=[1, 2], average="macro", zero_division=0)
    precision_riesgo = precision_score(y, y_pred, labels=[1, 2], average="macro", zero_division=0)
    f1_macro = f1_score(y, y_pred, average="macro")
    recall_all = recall_score(y, y_pred, average=None)
    precision_all = precision_score(y, y_pred, average=None)
    acc = (y == y_pred).mean()

    # % de días con aviso (clase 1 o 2)
    avisos_pct = ((y_pred == 1) | (y_pred == 2)).mean()

    # Matriz de confusión
    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])

    report = classification_report(y, y_pred, labels=[0, 1, 2],
                                   target_names=["seguro", "precaución", "peligro"],
                                   zero_division=0, output_dict=True)

    return {
        "clase": clase,
        "modelo": f"{'XGBoost' if clase == 'calor' else 'RandomForest'}_{clase}",
        "n_muestras": len(y),
        "recall_riesgo": round(recall_riesgo, 4),
        "precision_riesgo": round(precision_riesgo, 4),
        "f1_macro": round(f1_macro, 4),
        "accuracy": round(acc, 4),
        "recall_seguro": round(recall_all[0], 4),
        "recall_precaucion": round(recall_all[1], 4),
        "recall_peligro": round(recall_all[2], 4),
        "precision_seguro": round(precision_all[0], 4),
        "precision_precaucion": round(precision_all[1], 4),
        "precision_peligro": round(precision_all[2], 4),
        "pct_avisos": round(avisos_pct, 4),
        "confusion_matrix": cm.tolist(),
    }


# ── 2. Uso real (DB) ─────────────────────────────────────────────────

def compute_usage_metrics() -> dict:
    """Métricas de uso desde la base de datos."""
    if not DB_PATH.exists():
        return {"error": "DB no encontrada", "db_path": str(DB_PATH)}

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Total de consultas
    cur.execute("SELECT COUNT(*) FROM historial_consultas")
    total_consultas = cur.fetchone()[0]

    # Consultas por tipo_riesgo
    cur.execute("SELECT tipo_riesgo, COUNT(*) FROM historial_consultas GROUP BY tipo_riesgo")
    consultas_por_tipo = dict(cur.fetchall())

    # Perfiles únicos que han consultado
    cur.execute("SELECT COUNT(DISTINCT perfil_id) FROM historial_consultas WHERE perfil_id IS NOT NULL")
    perfiles_con_consulta = cur.fetchone()[0]

    # Total de perfiles creados
    cur.execute("SELECT COUNT(*) FROM perfiles")
    total_perfiles = cur.fetchone()[0]

    # Perfiles con datos de personalización (edad Y sexo)
    cur.execute("SELECT COUNT(*) FROM perfiles WHERE edad IS NOT NULL AND sexo IS NOT NULL")
    perfiles_personalizados = cur.fetchone()[0]

    # % con perfil personalizado
    pct_personalizado = round(perfiles_personalizados / total_perfiles, 4) if total_perfiles > 0 else 0

    # Perfiles con telegram_chat_id (usuarios del bot)
    cur.execute("SELECT COUNT(*) FROM perfiles WHERE telegram_chat_id IS NOT NULL")
    perfiles_telegram = cur.fetchone()[0]

    # Rango temporal
    cur.execute("SELECT MIN(created_at), MAX(created_at) FROM historial_consultas")
    fecha_min, fecha_max = cur.fetchone()

    # Dias con actividad
    cur.execute("SELECT COUNT(DISTINCT DATE(created_at)) FROM historial_consultas")
    dias_con_actividad = cur.fetchone()[0]

    # Provincias consultadas
    cur.execute("SELECT provincia, COUNT(*) FROM historial_consultas GROUP BY provincia ORDER BY COUNT(*) DESC")
    provincias = cur.fetchall()

    # Rutinas configuradas
    cur.execute("SELECT COUNT(*) FROM rutinas")
    rutinas = cur.fetchone()[0]

    # Avisos configurados
    cur.execute("SELECT COUNT(*) FROM avisos_config")
    avisos_config = cur.fetchone()[0]

    conn.close()

    return {
        "total_consultas": total_consultas,
        "consultas_por_tipo": consultas_por_tipo,
        "perfiles_con_consulta": perfiles_con_consulta,
        "total_perfiles": total_perfiles,
        "perfiles_personalizados": perfiles_personalizados,
        "pct_perfil_personalizado": pct_personalizado,
        "perfiles_telegram": perfiles_telegram,
        "rango_temporal": {"desde": fecha_min, "hasta": fecha_max},
        "dias_con_actividad": dias_con_actividad,
        "provincias_top5": provincias[:5],
        "rutinas_configuradas": rutinas,
        "avisos_configurados": avisos_config,
    }


# ── 3. Tráfico por canal (logs) ──────────────────────────────────────

def parse_log_counts(log_path: Path) -> dict:
    """Cuenta requests HTTP por endpoint en un log de uvicorn."""
    if not log_path.exists():
        return {"file": str(log_path), "exists": False}

    text = log_path.read_text(errors="replace")
    # Pattern: "POST /api/predict HTTP/1.1" 200 OK
    pattern = r'"(GET|POST|PUT|DELETE)\s+(/\S+)\s+HTTP/\S+"\s+(\d{3})'
    matches = re.findall(pattern, text)

    endpoints = Counter()
    status_codes = Counter()
    for method, endpoint, status in matches:
        # Normalize: strip query params
        base = endpoint.split("?")[0]
        endpoints[f"{method} {base}"] += 1
        status_codes[status] += 1

    return {
        "file": str(log_path.name),
        "exists": True,
        "total_requests": len(matches),
        "endpoints": dict(endpoints.most_common(20)),
        "status_codes": dict(status_codes),
    }


def compute_channel_metrics() -> dict:
    """Métricas de tráfico por canal (bot, web, MCP)."""
    bot = parse_log_counts(LOGS_DIR / "bot.log")
    web = parse_log_counts(LOGS_DIR / "web.log")
    mcp = parse_log_counts(LOGS_DIR / "mcp.log")
    return {"bot": bot, "web": web, "mcp": mcp}


# ── 4. Cobertura ──────────────────────────────────────────────────────

def compute_coverage() -> dict:
    """Cobertura: provincias × días con aviso fiable."""
    # Intentar calcular desde el dataset procesado
    coverage = {}
    for clase in ["calor", "frio"]:
        ds_path = PROCESSED_DIR / f"dataset_{clase}_labeled.parquet"
        if ds_path.exists():
            ds = pd.read_parquet(ds_path)
            n_provincias = ds["provincia"].nunique() if "provincia" in ds.columns else 0
            n_dias = ds["fecha"].nunique() if "fecha" in ds.columns else 0
            coverage[clase] = {
                "provincias": n_provincias,
                "dias": n_dias,
                "provincias_x_dias": n_provincias * n_dias,
                "fuente": str(ds_path.name),
            }
        else:
            coverage[clase] = {"error": f"No encontrado {ds_path.name}"}
    return coverage


# ── 5. Anticipación ──────────────────────────────────────────────────

def compute_anticipation() -> dict:
    """Anticipación: ¿se usan fuentes de forecast (Open-Meteo)?"""
    # Buscar en el código si hay llamadas a Open-Meteo
    readme_path = ROOT / "README.md"
    has_forecast = False
    forecast_source = None
    if readme_path.exists():
        text = readme_path.read_text()
        if "Open-Meteo" in text or "open-meteo" in text.lower():
            has_forecast = True
            forecast_source = "README.md menciona Open-Meteo como fuente de forecast"

    return {
        "tiene_forecast": has_forecast,
        "fuente": forecast_source,
        "medible_hoy": False,
        "nota": "La anticipación requiere un sistema de pronóstico activo (Open-Meteo) "
                "que genere predicciones futuras. Actualmente el pipeline usa datos históricos "
                "ERA5, no forecast en producción. Medir la anticipación real requiere "
                "implementar la ingesta de Open-Meteo y comparar predicciones vs realidad "
                "con suficiente historial.",
    }


# ── Comparativa con PRD ──────────────────────────────────────────────

def build_comparison_table(model_calor: dict, model_frio: dict) -> list:
    """Tabla comparativa: valores medidos vs líneas base del PRD."""
    rows = []

    def add_row(met, medido, prd_key, unit=""):
        prd_val = PRD_BASELINES.get(prd_key, None)
        if prd_val is not None:
            diff = medido - prd_val
            status = "✅ por encima" if diff >= 0 else "❌ por debajo"
            rows.append({
                "métrica": met,
                "valor_medido": f"{medido:.4f}{unit}",
                "linea_base_PRD": f"{prd_val:.4f}{unit}",
                "diferencia": f"{diff:+.4f}{unit}",
                "estado": status,
            })
        else:
            rows.append({
                "métrica": met,
                "valor_medido": f"{medido:.4f}{unit}",
                "linea_base_PRD": "No definida",
                "diferencia": "—",
                "estado": "📏 sin baseline",
            })

    add_row("Recall_riesgo (calor)", model_calor["recall_riesgo"], "recall_riesgo_calor")
    add_row("Recall_riesgo (frío)", model_frio["recall_riesgo"], "recall_riesgo_frio")
    add_row("Recall peligro (calor)", model_calor["recall_peligro"], None)
    add_row("Recall peligro (frío)", model_frio["recall_peligro"], None)
    add_row("Precisión_riesgo (calor)", model_calor["precision_riesgo"], None)
    add_row("Precisión_riesgo (frío)", model_frio["precision_riesgo"], None)
    add_row("F1_macro (calor)", model_calor["f1_macro"], None)
    add_row("F1_macro (frío)", model_frio["f1_macro"], None)
    add_row("% avisos (calor)", model_calor["pct_avisos"] * 100, None, unit="%")
    add_row("% avisos (frío)", model_frio["pct_avisos"] * 100, None, unit="%")

    return rows


# ── Qué falta ────────────────────────────────────────────────────────

def what_is_missing(usage: dict, channels: dict, coverage: dict, anticipation: dict) -> list:
    """Documenta qué no se puede medir hoy."""
    missing = []

    # Anticipación
    if not anticipation.get("medible_hoy"):
        missing.append({
            "métrica": "Anticipación (días de antelación)",
            "qué_falta": "Pipeline de ingesta de Open-Meteo para generar predicciones futuras. "
                         "Histórico de predicciones vs realidad para calcular antelación real.",
            "pasos": [
                "Implementar ingesta Open-Meteo (DATA-001 o similar)",
                "Generar predicciones diarias y guardarlas en BD",
                "Tras N meses, comparar predicciones con datos MoMo reales",
                "Calcular días de antelación promedio",
            ],
        })

    # Cobertura
    if "error" in str(coverage.get("calor", "")):
        missing.append({
            "métrica": "Cobertura (provincias × días)",
            "qué_falta": "Dataset procesado no contiene columnas 'provincia' y 'date' en parquet. "
                         "O los datos de producción aún no cubren todas las provincias.",
            "pasos": [
                "Verificar que los parquets incluyen provincia y date",
                "Si faltan, regenerar dataset con make_dataset incluyendo metadatos",
                "Confirmar qué provincias están cubiertas por el modelo actual",
            ],
        })

    # Uso real: logs incompletos
    if channels.get("bot", {}).get("total_requests", 0) < 10:
        missing.append({
            "métrica": "Uso real — Bot de Telegram",
            "qué_falta": "Logs del bot muy escasos (posiblemente solo en pruebas). "
                         "No hay suficiente tráfico para métricas de uso.",
            "pasos": [
                "Desplegar bot en producción y generar tráfico real",
                "Configurar logging persistente con métricas por usuario",
                "Opcional: analytics por chat_id para uso diario/semanal",
            ],
        })

    if channels.get("web", {}).get("total_requests", 0) < 20:
        missing.append({
            "métrica": "Uso real — Web",
            "qué_falta": "Log web registrado (post/predict, chat) pero es local/dev.",
            "pasos": [
                "Desplegar web en servidor accesible",
                "Contar usuarios únicos reales vs localhost",
            ],
        })

    if channels.get("mcp", {}).get("total_requests", 0) < 5:
        missing.append({
            "métrica": "Uso real — API/MCP",
            "qué_falta": "MCP ha tenido 2-3 requests (pruebas).",
            "pasos": [
                "Integrar MCP con asistentes externos (Claude, etc.)",
                "Medir requests por sesión de usuario",
            ],
        })

    # % con perfil personalizado
    if usage.get("pct_perfil_personalizado", 0) < 0.5:
        missing.append({
            "métrica": "% con perfil personalizado",
            "qué_falta": f"Solo {usage.get('perfiles_personalizados', 0)}/{usage.get('total_perfiles', 0)} "
                         f"perfiles ({usage.get('pct_perfil_personalizado', 0)*100:.1f}%) tienen edad+sexo. "
                         "La mayoría de consultas son anónimas (perfil_id=None).",
            "pasos": [
                "Implementar flujo obligatorio de onboarding en bot/web",
                "Requerir edad y sexo antes de la primera predicción",
                "Prompt de personalización en cada interacción",
            ],
        })

    return missing


# ── Output ────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Mide métricas de éxito del PRD")
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    results = {}

    # 1. Métricas de modelo
    print_section("1. MÉTRICAS DE MODELO (test set)")
    model_calor = compute_model_metrics("calor")
    model_frio = compute_model_metrics("frio")
    results["modelo_calor"] = model_calor
    results["modelo_frio"] = model_frio

    for m in [model_calor, model_frio]:
        print(f"\n--- {m['modelo']} ---")
        print(f"  Muestras test:      {m['n_muestras']}")
        print(f"  Recall_riesgo:      {m['recall_riesgo']}")
        print(f"  Precisión_riesgo:   {m['precision_riesgo']}")
        print(f"  F1_macro:           {m['f1_macro']}")
        print(f"  Accuracy:           {m['accuracy']}")
        print(f"  Recall seguro:      {m['recall_seguro']}")
        print(f"  Recall precaución:  {m['recall_precaucion']}")
        print(f"  Recall peligro:     {m['recall_peligro']}")
        print(f"  % días con aviso:   {m['pct_avisos']*100:.1f}%")
        print(f"  Matriz confusión:")
        cm = m["confusion_matrix"]
        print(f"                  pred_seg  pred_prec  pred_pelig")
        for i, (label, row) in enumerate(zip(["real_seg", "real_prec", "real_pelig"], cm)):
            print(f"    {label:12s}  {row[0]:>8d}  {row[1]:>9d}  {row[2]:>10d}")

    # 2. Uso real
    print_section("2. USO REAL (base de datos)")
    usage = compute_usage_metrics()
    results["uso"] = usage

    if "error" not in usage:
        print(f"  Total consultas:          {usage['total_consultas']}")
        print(f"  Por tipo calor/frío:      {usage['consultas_por_tipo']}")
        print(f"  Perfiles creados:         {usage['total_perfiles']}")
        print(f"  Perfiles personalizados:  {usage['perfiles_personalizados']}/{usage['total_perfiles']}")
        print(f"  % con perfil personalizado: {usage['pct_perfil_personalizado']*100:.1f}%")
        print(f"  Perfiles con Telegram:    {usage['perfiles_telegram']}")
        print(f"  Perfiles con consulta:    {usage['perfiles_con_consulta']}")
        print(f"  Rutinas configuradas:     {usage['rutinas_configuradas']}")
        print(f"  Avisos diarios configurados: {usage['avisos_configurados']}")
        print(f"  Rango temporal:           {usage['rango_temporal']['desde']} → {usage['rango_temporal']['hasta']}")
        print(f"  Días con actividad:       {usage['dias_con_actividad']}")
        print(f"  Provincias top 5:")
        for prov, count in usage["provincias_top5"]:
            print(f"    {prov:20s}  {count} consultas")
    else:
        print(f"  ERROR: {usage['error']}")

    # 3. Tráfico por canal
    print_section("3. TRÁFICO POR CANAL (logs)")
    channels = compute_channel_metrics()
    results["canales"] = channels

    for canal_name, canal_data in channels.items():
        print(f"\n--- {canal_name.upper()} ({canal_data['file']}) ---")
        if canal_data.get("exists"):
            print(f"  Requests totales:  {canal_data['total_requests']}")
            print(f"  Status codes:      {canal_data['status_codes']}")
            print(f"  Endpoints:")
            for ep, count in canal_data["endpoints"].items():
                print(f"    {ep:40s}  {count}")
        else:
            print(f"  Log no encontrado")

    # 4. Cobertura
    print_section("4. COBERURA (provincias × días)")
    coverage = compute_coverage()
    results["cobertura"] = coverage
    for clase, data in coverage.items():
        print(f"\n--- {clase.upper()} ---")
        for k, v in data.items():
            print(f"  {k}: {v}")

    # 5. Anticipación
    print_section("5. ANTICIPACIÓN")
    anticipation = compute_anticipation()
    results["anticipacion"] = anticipation
    print(f"  Tiene forecast:    {anticipation['tiene_forecast']}")
    print(f"  Fuente:            {anticipation['fuente']}")
    print(f"  Medible hoy:       {anticipation['medible_hoy']}")
    print(f"  Nota:              {anticipation['nota']}")

    # 6. Comparativa con PRD
    print_section("6. COMPARATIVA: VALORES MEDIDOS vs PRD BASELINES")
    comparison = build_comparison_table(model_calor, model_frio)
    results["comparativa"] = comparison

    # Print table
    header = f"{'Métrica':<30s} {'Medido':>12s} {'PRD Baseline':>14s} {'Δ':>12s} {'Estado':>18s}"
    print(f"\n{header}")
    print("-" * len(header))
    for row in comparison:
        print(f"{row['métrica']:<30s} {row['valor_medido']:>12s} {row['linea_base_PRD']:>14s} {row['diferencia']:>12s} {row['estado']:>18s}")

    # 7. Qué falta
    print_section("7. QUÉ FALTA PARA MEDIR TODO")
    missing = what_is_missing(usage, channels, coverage, anticipation)
    results["falta"] = missing

    if missing:
        for item in missing:
            print(f"\n  ❌ {item['métrica']}")
            print(f"     Falta: {item['qué_falta']}")
            print(f"     Pasos:")
            for paso in item["pasos"]:
                print(f"       - {paso}")
    else:
        print("  ✅ Todas las métricas se pueden medir con los datos actuales.")

    # JSON output
    if args.json:
        print(f"\n{'='*70}")
        print("  JSON OUTPUT")
        print(f"{'='*70}")
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
