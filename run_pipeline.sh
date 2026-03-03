#!/bin/bash
# Pipeline completo: PDFs → Figuras → Extracción → Dataset v2
# Uso: ./run_pipeline.sh [--paper P01] [--test]

set -e

source .env

PDF_DIR=${PDF_SOURCE_DIR:-"/mnt/agent-workspace/Investigación/Cholula-Papers/"}
FIGURES_DIR="./figures"
EXTRACTED_JSON="./data/extracted_figures.json"
V1_CSV="./data/raw_extracted.csv"
V2_CSV="./data/dataset_v2.csv"

PAPER_FILTER=""
if [ "$1" = "--paper" ]; then
  PAPER_FILTER="--paper $2"
fi

echo "=========================================="
echo "🔬 Cholula-ML: Pipeline Extracción Figuras"
echo "=========================================="
echo "PDF dir: $PDF_DIR"
echo ""

# Fase 2: Extraer figuras
echo "📂 FASE 2: Extrayendo figuras de PDFs..."
python3 nps_pipeline/extract_figures.py \
  --pdf-dir "$PDF_DIR" \
  --output-dir "$FIGURES_DIR" $PAPER_FILTER

echo ""
echo "🤖 FASE 3: Extrayendo datos con Claude Vision..."
python3 nps_pipeline/extract_nps_data.py \
  --figures-dir "$FIGURES_DIR" \
  --output "$EXTRACTED_JSON" \
  --delay 2.0 $PAPER_FILTER

echo ""
echo "🔗 FASE 4: Mergeando con dataset v1..."
python3 nps_pipeline/merge_dataset.py \
  --v1 "$V1_CSV" \
  --figures "$EXTRACTED_JSON" \
  --output "$V2_CSV"

echo ""
echo "✅ Pipeline completado!"
echo "Dataset v2: $V2_CSV"
echo "Reporte: ./data/merge_report.md"
