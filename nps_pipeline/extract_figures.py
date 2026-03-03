"""
Fase 2: Extracción de figuras de PDFs usando pdfimages (poppler).

Usa pdfimages (parte de poppler-utils, ya instalado en el sistema) para extraer
todas las imágenes incrustadas en cada PDF.

Uso:
    python nps_pipeline/extract_figures.py --pdf-dir /path/to/pdfs --output-dir ./figures
    # Solo un paper para testing:
    python nps_pipeline/extract_figures.py --pdf-dir /path/to/pdfs --output-dir ./figures --paper P01

Output:
    figures/
        P01/
            P01-000.png
            P01-001.png
            ...
        P02/
            ...
        metadata.json  ← índice de todas las figuras extraídas
"""

import subprocess
import os
import json
import argparse
import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MIN_WIDTH_PX = 150
MIN_HEIGHT_PX = 150

# Mapeo de prefijo de filename → paper_id
PDF_TO_PAPER_ID = {
    "01_": "P01", "02_": "P02", "03_": "P03",
    "04_": "P04", "05_": "P05", "07_": "P07",
}

def get_paper_id(filename: str) -> str:
    for prefix, pid in PDF_TO_PAPER_ID.items():
        if filename.startswith(prefix):
            return pid
    return Path(filename).stem[:3].upper()


def extract_figures_from_pdf(pdf_path: Path, output_dir: Path, paper_id: str) -> list[dict]:
    """Extrae imágenes de un PDF con pdfimages y filtra las relevantes."""
    paper_dir = output_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(paper_dir / paper_id)

    # pdfimages -all extrae todas las imágenes en su formato original
    result = subprocess.run(
        ["pdfimages", "-all", str(pdf_path), prefix],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        logger.error(f"pdfimages error: {result.stderr}")
        return []

    # Recopilar imágenes generadas
    extracted_files = sorted(paper_dir.glob(f"{paper_id}-*"))
    figures = []

    for img_path in extracted_files:
        # Intentar abrir con PIL para verificar dimensiones
        try:
            img = Image.open(str(img_path)).convert("RGB")
            w, h = img.size

            if w < MIN_WIDTH_PX or h < MIN_HEIGHT_PX:
                img_path.unlink()  # Borrar imágenes pequeñas (logos, íconos)
                continue

            # Convertir a PNG si no lo es
            png_path = img_path.with_suffix(".png")
            if img_path.suffix.lower() != ".png":
                img.save(str(png_path), "PNG")
                img_path.unlink()
                img_path = png_path

            fig_idx = int(img_path.stem.split("-")[-1]) if "-" in img_path.stem else 0

            meta = {
                "paper_id": paper_id,
                "pdf": pdf_path.name,
                "figure_index": fig_idx,
                "filename": img_path.name,
                "path": str(img_path),
                "width_px": w,
                "height_px": h,
                "aspect_ratio": round(w / h, 2),
                "type_guess": None,
                "extraction_status": "pending",
            }
            figures.append(meta)
            logger.info(f"  ✅ {img_path.name} ({w}×{h}px)")

        except Exception as e:
            logger.warning(f"  ⚠️ {img_path.name}: {e}")
            continue

    logger.info(f"📄 {paper_id}: {len(figures)} figuras válidas extraídas")
    return figures


def run(pdf_dir: str, output_dir: str, paper_filter: str = None) -> None:
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.error(f"No se encontraron PDFs en {pdf_dir}")
        return

    if paper_filter:
        target_prefix = paper_filter.replace("P", "0").zfill(2) + "_"
        pdfs = [p for p in pdfs if p.name.startswith(target_prefix[:2])]
        logger.info(f"Filtrado a: {[p.name for p in pdfs]}")

    all_figures = []
    for pdf_path in pdfs:
        paper_id = get_paper_id(pdf_path.name)
        logger.info(f"\n📂 {pdf_path.name} → {paper_id}")
        figs = extract_figures_from_pdf(pdf_path, output_dir, paper_id)
        all_figures.extend(figs)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_figures, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ TOTAL: {len(all_figures)} figuras de {len(pdfs)} PDFs")
    logger.info(f"📁 Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=os.getenv("PDF_SOURCE_DIR", "./pdfs"))
    parser.add_argument("--output-dir", default=os.getenv("FIGURES_OUTPUT_DIR", "./figures"))
    parser.add_argument("--paper", default=None, help="Filtrar por paper (P01, P02...)")
    args = parser.parse_args()
    run(args.pdf_dir, args.output_dir, args.paper)
