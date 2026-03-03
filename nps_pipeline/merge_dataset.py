"""
Fase 4: Integra los datos extraídos de figuras con el dataset existente (v1).

Lee:
  - data/raw_extracted.csv (dataset_v1, 50 rows, datos de texto)
  - data/extracted_figures.json (datos extraídos de figuras con Claude Vision)

Produce:
  - data/dataset_v2.csv: dataset enriquecido
  - data/merge_report.md: reporte de completitud por columna

Lógica de merge:
  - Match por paper_id + np_composition + cell_line (cuando es posible)
  - Para cada data_point extraído, busca la fila del dataset_v1 más compatible
  - Si hay match → actualiza la fila con el nuevo valor (si estaba vacío)
  - Si no hay match → crea fila nueva con paper_id y datos disponibles
"""

import pandas as pd
import json
import os
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Mapeo variable extraída → columna del dataset v1
VAR_TO_COLUMN = {
    "pdi": "pdi",
    "ic50_ug_ml": "ic50_ug_ml",
    "ic50_um": "ic50_um",
    "viability_pct": "viability_pct",
    "ros_pct": "ros_pct",
    "mic_ug_ml": "mic_ug_ml",
    "lspr_nm": "lspr_nm",
}


def load_figures_data(figures_json: str) -> list[dict]:
    """Carga y filtra solo los data points extraíbles con datos reales."""
    with open(figures_json, encoding="utf-8") as f:
        figures = json.load(f)
    
    data_points = []
    for fig in figures:
        if fig.get("extraction_status") != "done":
            continue
        extracted = fig.get("extracted_data", {})
        if not extracted.get("extractable", False):
            continue
        
        for dp in extracted.get("data_points", []):
            if dp.get("y_value") is None:
                continue
            if dp.get("variable_extracted") not in VAR_TO_COLUMN:
                continue
            if dp.get("confidence") == "low":
                continue
            
            data_points.append({
                "paper_id": fig["paper_id"],
                "figure_file": fig["filename"],
                "variable": dp["variable_extracted"],
                "column": VAR_TO_COLUMN[dp["variable_extracted"]],
                "x_value": dp.get("x_value"),
                "y_value": dp["y_value"],
                "label": dp.get("label", ""),
                "confidence": dp.get("confidence", "medium"),
                "np_compositions": extracted.get("np_compositions_mentioned", []),
                "cell_lines": extracted.get("cell_lines_mentioned", []),
                "bacteria": extracted.get("bacteria_mentioned", []),
                "chart_type": extracted.get("chart_type", "unknown"),
            })
    
    logger.info(f"📊 {len(data_points)} data points válidos de figuras")
    return data_points


def simple_merge(df_v1: pd.DataFrame, data_points: list[dict]) -> pd.DataFrame:
    """
    Merge simple: intenta llenar valores nulos en df_v1 con datos de figuras.
    Para cada data_point, busca la fila de mismo paper_id con columna vacía.
    """
    df = df_v1.copy()
    new_rows = []
    filled = 0
    added = 0

    for dp in data_points:
        pid = dp["paper_id"]
        col = dp["column"]
        val = dp["y_value"]

        # Buscar fila del mismo paper_id con esa columna vacía
        mask_paper = df["paper_id"] == pid
        mask_empty = df[col].isna() if col in df.columns else pd.Series(False, index=df.index)
        
        candidates = df[mask_paper & mask_empty] if col in df.columns else pd.DataFrame()
        
        if len(candidates) > 0:
            # Tomar el primer candidato
            idx = candidates.index[0]
            df.loc[idx, col] = val
            df.loc[idx, "data_source"] = "figure_extracted"
            filled += 1
        else:
            # Crear fila nueva con la info disponible
            new_row = {"paper_id": pid}
            if col in df.columns:
                new_row[col] = val
            
            if dp["cell_lines"]:
                new_row["cell_line"] = dp["cell_lines"][0]
            if dp["np_compositions"]:
                new_row["np_composition_raw"] = dp["np_compositions"][0]
            
            new_row["data_source"] = "figure_extracted_new"
            new_row["figure_source"] = dp["figure_file"]
            new_rows.append(new_row)
            added += 1

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df = pd.concat([df, df_new], ignore_index=True)

    logger.info(f"✅ {filled} celdas llenadas en filas existentes")
    logger.info(f"➕ {added} filas nuevas creadas desde figuras")
    return df


def generate_report(df_v1: pd.DataFrame, df_v2: pd.DataFrame, output_path: str) -> None:
    """Genera reporte de completitud antes vs después."""
    target_cols = ["pdi", "ic50_ug_ml", "viability_pct", "ros_pct", "mic_ug_ml", "lspr_nm"]
    
    lines = ["# Reporte de Merge — Dataset v2\n",
             f"- Filas v1: {len(df_v1)}\n",
             f"- Filas v2: {len(df_v2)}\n",
             f"- Filas nuevas: {len(df_v2) - len(df_v1)}\n\n",
             "## Completitud por columna\n\n",
             "| Columna | v1 (%) | v2 (%) | Mejora |\n",
             "|---------|--------|--------|--------|\n"]
    
    for col in target_cols:
        if col not in df_v1.columns:
            pct_v1 = 0.0
        else:
            pct_v1 = df_v1[col].notna().mean() * 100
        
        if col not in df_v2.columns:
            pct_v2 = 0.0
        else:
            pct_v2 = df_v2[col].notna().mean() * 100
        
        mejora = f"+{pct_v2 - pct_v1:.1f}%" if pct_v2 > pct_v1 else "—"
        lines.append(f"| {col} | {pct_v1:.1f}% | {pct_v2:.1f}% | {mejora} |\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    
    logger.info(f"📄 Reporte: {output_path}")


def run(v1_path: str, figures_json: str, output_v2: str) -> None:
    # Cargar dataset v1
    df_v1 = pd.read_csv(v1_path)
    logger.info(f"📂 Dataset v1: {len(df_v1)} filas × {len(df_v1.columns)} columnas")

    # Cargar data points de figuras
    data_points = load_figures_data(figures_json)

    if not data_points:
        logger.warning("No hay data points para mergear")
        return

    # Merge
    df_v2 = simple_merge(df_v1, data_points)
    
    # Guardar
    df_v2.to_csv(output_v2, index=False)
    logger.info(f"💾 Dataset v2 guardado: {output_v2} ({len(df_v2)} filas)")

    # Reporte
    report_path = str(Path(output_v2).parent / "merge_report.md")
    generate_report(df_v1, df_v2, report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge figura-datos con dataset v1")
    parser.add_argument("--v1", default="./data/raw_extracted.csv", help="Dataset v1 (CSV)")
    parser.add_argument("--figures", default="./data/extracted_figures.json", help="JSON de figuras extraídas")
    parser.add_argument("--output", default="./data/dataset_v2.csv", help="Dataset v2 de salida")
    args = parser.parse_args()
    run(args.v1, args.figures, args.output)
