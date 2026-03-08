"""
Fase 3 OPTIMIZADA: Extracción de datos con filtrado por dimensiones.
Solo procesa figuras con w>500, h>400, aspect_ratio 0.5-2.5.
Evita TEM/SEM cuadrados pequeños.
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
  "cell_lines_mentioned": ["MCF-7", "HeLa", "H9c2"],
  "bacteria_mentioned": ["E. coli", "S. aureus"],
  "notes": "any important context about the figure",
  "extractable": true,
  "reason_if_not_extractable": "e.g., figure is a TEM image, schematic, or fluorescence microscopy"
}

If the figure is a TEM/SEM image, microscopy image, or schematic (not a chart with quantitative data), set extractable=false.

Be precise with values. If a bar chart shows IC50=12.5 µg/mL for AgAu 50:50, report x_value=null, y_value=12.5, label="AgAu 50:50", variable_extracted="ic50_ug_ml".
"""


def encode_image(image_path: str) -> tuple:
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower()
    media_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    media_type = media_types.get(ext, "image/png")
    return image_data, media_type


def extract_from_figure(client, fig_meta: dict) -> dict:
    fig_path = fig_meta["path"]
    fig_name = fig_meta["filename"]

    # Resolve path relative to project root if needed
    if not Path(fig_path).exists():
        alt_path = Path("/home/jorge/workspace/plot-digitizer-nps") / fig_path
        if alt_path.exists():
            fig_path = str(alt_path)
        else:
            return {**fig_meta, "extraction_status": "error", "error": f"File not found: {fig_path}"}

    try:
        image_data, media_type = encode_image(fig_path)
    except Exception as e:
        return {**fig_meta, "extraction_status": "error", "error": f"Could not encode: {e}"}

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": image_data},
                    },
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }],
        )

        raw_text = response.content[0].text.strip()

        # Strip markdown code blocks if present
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            raw_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            extracted = json.loads(raw_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if json_match:
                try:
                    extracted = json.loads(json_match.group())
                except:
                    extracted = {"error": "JSON parse failed", "raw": raw_text[:500], "extractable": False}
            else:
                extracted = {"error": "No JSON found", "raw": raw_text[:500], "extractable": False}

        result = {
            **fig_meta,
            "extraction_status": "done",
            "extracted_data": extracted,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
        }

        is_extractable = extracted.get("extractable", True)
        dp_count = len(extracted.get("data_points", []))
        logger.info(f"  ✅ {fig_name}: {'extractable' if is_extractable else 'NOT extractable'}, {dp_count} data points")
        return result

    except Exception as e:
        logger.error(f"  ❌ {fig_name}: API error: {e}")
        return {**fig_meta, "extraction_status": "error", "error": str(e)}


def run(figures_dir: str, output_path: str, paper_filter: str = None, delay: float = 2.0,
        min_w: int = 500, min_h: int = 400, min_ar: float = 0.5, max_ar: float = 2.5) -> dict:
    figures_dir = Path(figures_dir)
    metadata_path = figures_dir / "metadata.json"

    if not metadata_path.exists():
        logger.error(f"metadata.json no encontrado en {figures_dir}")
        return {}

    with open(metadata_path, encoding="utf-8") as f:
        all_figures = json.load(f)

    # Filter by paper
    if paper_filter:
        all_figures = [f for f in all_figures if f["paper_id"] == paper_filter.upper()]
        logger.info(f"Paper {paper_filter}: {len(all_figures)} figuras totales")

    # Filter by dimensions (skip tiny/square TEM images)
    filtered = []
    skipped = []
    for fig in all_figures:
        w = fig.get("width_px", 0)
        h = fig.get("height_px", 0)
        ar = fig.get("aspect_ratio", 0)
        if w > min_w and h > min_h and min_ar <= ar <= max_ar:
            filtered.append(fig)
        else:
            skipped.append(fig["filename"])

    logger.info(f"  Filtradas: {len(filtered)} ✅  Saltadas (TEM/small): {len(skipped)} ❌")
    if skipped:
        logger.info(f"  Saltadas: {skipped}")

    if not filtered:
        logger.warning("No hay figuras para procesar después del filtro")
        return {"processed": 0, "extractable": 0, "data_points": 0}

    # Init client
    api_key = os.getenv("API_KEY_ANTHROPIC")
    if not api_key:
        raise ValueError("API_KEY_ANTHROPIC no configurada en .env")
    client = anthropic.Anthropic(api_key=api_key)

    results = []
    total = len(filtered)
    logger.info(f"\n🔬 Procesando {total} figuras con Claude Vision...\n")

    # Load existing results if any (incremental)
    if Path(output_path).exists():
        with open(output_path) as f:
            existing = json.load(f)
        done_names = {r["filename"] for r in existing if r.get("extraction_status") == "done"}
        results = existing
        logger.info(f"  Retomando: {len(done_names)} ya procesadas")
    else:
        done_names = set()

    for i, fig_meta in enumerate(filtered):
        if fig_meta["filename"] in done_names:
            logger.info(f"[{i+1}/{total}] SKIP (ya procesada): {fig_meta['filename']}")
            continue

        logger.info(f"[{i+1}/{total}] {fig_meta['filename']}")
        result = extract_from_figure(client, fig_meta)
        results.append(result)

        # Incremental save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if i < total - 1:
            time.sleep(delay)

    # Stats
    done = [r for r in results if r.get("extraction_status") == "done"]
    extractable_results = [r for r in done if r.get("extracted_data", {}).get("extractable", False)]
    total_dp = sum(len(r.get("extracted_data", {}).get("data_points", [])) for r in done)
    total_tokens = sum(r.get("tokens_used", 0) for r in done)

    logger.info(f"\n📊 RESUMEN {paper_filter or 'ALL'}:")
    logger.info(f"  Procesadas: {len(done)}/{total}")
    logger.info(f"  Con datos: {len(extractable_results)}")
    logger.info(f"  Data points: {total_dp}")
    logger.info(f"  Tokens usados: {total_tokens:,}")
    logger.info(f"  Output: {output_path}")

    return {
        "paper": paper_filter,
        "processed": len(done),
        "extractable": len(extractable_results),
        "data_points": total_dp,
        "tokens_used": total_tokens,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures-dir", default="./figures")
    parser.add_argument("--output", default="./data/extracted_figures.json")
    parser.add_argument("--paper", default=None)
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    run(args.figures_dir, args.output, args.paper, args.delay)
