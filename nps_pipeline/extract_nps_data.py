"""
Fase 3: Extracción de datos de nanopartículas desde figuras usando Claude Vision.

Para cada figura en metadata.json, llama a Claude claude-opus-4-6 con un prompt
especializado para extraer:
  - PDI (polydispersity index)
  - IC50 (µg/mL o µM)
  - Viabilidad celular (%)
  - ROS (reactive oxygen species, % vs control)
  - MIC (minimum inhibitory concentration)
  - Tipo de gráfica detectado

Uso:
    python nps_pipeline/extract_nps_data.py --figures-dir ./figures --output ./data/extracted_figures.json
    python nps_pipeline/extract_nps_data.py --figures-dir ./figures --paper P01  # solo un paper
"""

import anthropic
import base64
import json
import os
import time
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Columnas objetivo del dataset Cholula-ML
NPS_TARGET_COLUMNS = [
    "pdi",           # polydispersity index (0-1)
    "ic50_ug_ml",    # IC50 en µg/mL
    "ic50_um",       # IC50 en µM (si reportado en µM)
    "viability_pct", # % viabilidad celular (0-100)
    "ros_pct",       # % ROS vs control (100+ = aumento)
    "mic_ug_ml",     # MIC en µg/mL
    "lspr_nm",       # peak LSPR en nm
]

EXTRACTION_PROMPT = """You are a scientific data extraction specialist for nanoparticle bioactivity research.

Analyze this figure from a scientific paper about metallic nanoparticles (AgNPs, AuNPs, SeNPs, bimetallic systems).

Your task: Extract quantitative data values from the figure.

**Target variables** (extract only those visible in the figure):
- PDI (polydispersity index): dimensionless value 0-1 from DLS measurements
- IC50: concentration causing 50% inhibition (in µg/mL or µM — note the units!)
- Cell viability (%): percentage of viable cells vs control (often at specific concentrations)
- ROS (%): reactive oxygen species relative to control (100% = same as control)
- MIC (µg/mL): minimum inhibitory concentration for antimicrobial data
- LSPR (nm): localized surface plasmon resonance peak wavelength

**Output format** (strict JSON, no markdown):
{
  "chart_type": "bar|scatter|line|table|size_distribution|dose_response|other",
  "x_axis": {"label": "...", "unit": "..."},
  "y_axis": {"label": "...", "unit": "..."},
  "data_points": [
    {
      "x_value": <number or null>,
      "y_value": <number or null>,
      "label": "description of this data point (e.g., NP composition, cell line)",
      "variable_extracted": "pdi|ic50_ug_ml|ic50_um|viability_pct|ros_pct|mic_ug_ml|lspr_nm|other",
      "confidence": "high|medium|low"
    }
  ],
  "np_compositions_mentioned": ["e.g., Ag50Au50", "AgNPs", "AuNPs"],
  "cell_lines_mentioned": ["MCF-7", "HeLa", "H9c2", etc.],
  "bacteria_mentioned": ["E. coli", "S. aureus", etc.],
  "notes": "any important context about the figure",
  "extractable": true/false,
  "reason_if_not_extractable": "e.g., figure is a TEM image, schematic, or fluorescence microscopy"
}

If the figure is a TEM/SEM image, microscopy image, or schematic (not a chart with quantitative data), set extractable=false.

Be precise with values. If a bar chart shows IC50=12.5 µg/mL for AgAu 50:50, report x_value=null, y_value=12.5, label="AgAu 50:50", variable_extracted="ic50_ug_ml".
"""


def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 for Claude API."""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(ext, "image/png")
    return image_data, media_type


def extract_from_figure(client: anthropic.Anthropic, fig_meta: dict) -> dict:
    """
    Llama a Claude Vision para extraer datos de una figura.
    
    Returns:
        Dict con los datos extraídos + metadata de la figura.
    """
    fig_path = fig_meta["path"]
    paper_id = fig_meta["paper_id"]
    fig_name = fig_meta["filename"]

    try:
        image_data, media_type = encode_image(fig_path)
    except Exception as e:
        return {**fig_meta, "extraction_status": "error", "error": f"Could not encode: {e}"}

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": EXTRACTION_PROMPT,
                        },
                    ],
                }
            ],
        )
        
        raw_text = response.content[0].text.strip()
        
        # Parse JSON
        try:
            extracted = json.loads(raw_text)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
            else:
                extracted = {"error": "JSON parse failed", "raw": raw_text[:500]}
        
        result = {
            **fig_meta,
            "extraction_status": "done",
            "extracted_data": extracted,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }
        
        extractable = extracted.get("extractable", True)
        dp_count = len(extracted.get("data_points", []))
        logger.info(f"  ✅ {fig_name}: {'extractable' if extractable else 'NOT extractable'}, {dp_count} data points")
        
        return result

    except Exception as e:
        logger.error(f"  ❌ {fig_name}: {e}")
        return {**fig_meta, "extraction_status": "error", "error": str(e)}


def run(figures_dir: str, output_path: str, paper_filter: str = None, delay: float = 2.0) -> None:
    """Pipeline principal de extracción."""
    figures_dir = Path(figures_dir)
    
    # Prefer candidates.json (pre-filtered charts) over metadata.json (all figures)
    candidates_path = figures_dir / "candidates.json"
    metadata_path = figures_dir / "metadata.json"
    
    if candidates_path.exists():
        source_path = candidates_path
        logger.info(f"Using candidates.json (pre-filtered charts)")
    elif metadata_path.exists():
        source_path = metadata_path
        logger.info(f"Using metadata.json (all figures — no candidates filter)")
    else:
        logger.error(f"Neither candidates.json nor metadata.json found in {figures_dir}")
        logger.error("Ejecuta primero: python nps_pipeline/extract_figures.py")
        return

    with open(source_path, encoding="utf-8") as f:
        all_figures = json.load(f)

    # Filtrar por paper si se especifica
    if paper_filter:
        all_figures = [f for f in all_figures if f["paper_id"] == paper_filter.upper()]
        logger.info(f"Filtrado a {len(all_figures)} figuras de {paper_filter}")

    if not all_figures:
        logger.warning("No hay figuras para procesar")
        return

    # Inicializar cliente
    api_key = os.getenv("API_KEY_ANTHROPIC")
    if not api_key:
        raise ValueError("API_KEY_ANTHROPIC no configurada")
    client = anthropic.Anthropic(api_key=api_key)

    results = []
    total = len(all_figures)
    logger.info(f"\n🔬 Extrayendo datos de {total} figuras con Claude Vision...\n")

    for i, fig_meta in enumerate(all_figures):
        logger.info(f"[{i+1}/{total}] {fig_meta['filename']}")
        result = extract_from_figure(client, fig_meta)
        results.append(result)

        # Guardar progreso incremental
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if i < total - 1:
            time.sleep(delay)  # Rate limiting

    # Estadísticas finales
    done = sum(1 for r in results if r["extraction_status"] == "done")
    extractable = sum(1 for r in results 
                     if r.get("extracted_data", {}).get("extractable", False))
    total_dp = sum(len(r.get("extracted_data", {}).get("data_points", [])) for r in results)
    
    logger.info(f"\n📊 RESUMEN:")
    logger.info(f"  Total figuras: {total}")
    logger.info(f"  Procesadas: {done}")
    logger.info(f"  Con datos extraíbles: {extractable}")
    logger.info(f"  Total data points: {total_dp}")
    logger.info(f"  Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extrae datos NPs de figuras con Claude Vision")
    parser.add_argument("--figures-dir", default="./figures",
                        help="Directorio con las figuras extraídas (de extract_figures.py)")
    parser.add_argument("--output", default="./data/extracted_figures.json",
                        help="Archivo JSON de salida")
    parser.add_argument("--paper", default=None,
                        help="Filtrar por paper_id (P01, P02...) para testing")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="Delay en segundos entre llamadas a API (default: 2)")
    args = parser.parse_args()
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args.figures_dir, args.output, args.paper, args.delay)
