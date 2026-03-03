"""
Extractor alternativo: Renderiza PÁGINAS completas de PDFs como imágenes.

Usar cuando pdfimages no encuentra imágenes incrustadas (papers con figuras vectoriales).
pdftoppm renderiza cada página del PDF a alta resolución, capturando tanto imágenes
raster como figuras vectoriales (EPS, SVG renderizadas).

Uso:
    python nps_pipeline/extract_pages.py --pdf-dir /path --output-dir ./pages
    # Después filtrar manualmente las páginas con figuras (o usar Claude para clasificar)
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

DPI = 200  # Mayor DPI = mejor calidad pero más lento
PDF_TO_PAPER_ID = {
    "01_": "P01", "02_": "P02", "03_": "P03",
    "04_": "P04", "05_": "P05", "07_": "P07",
}

def get_paper_id(filename: str) -> str:
    for prefix, pid in PDF_TO_PAPER_ID.items():
        if filename.startswith(prefix):
            return pid
    return Path(filename).stem[:3].upper()


def render_pdf_pages(pdf_path: Path, output_dir: Path, paper_id: str, dpi: int = DPI) -> list[dict]:
    """Renderiza todas las páginas de un PDF como PNG."""
    paper_dir = output_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(paper_dir / paper_id)

    result = subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), str(pdf_path), prefix],
        capture_output=True, text=True
    )

    if result.returncode != 0:
        logger.error(f"pdftoppm error: {result.stderr}")
        return []

    pages = sorted(paper_dir.glob(f"{paper_id}-*.png"))
    page_metas = []
    for pg_path in pages:
        try:
            img = Image.open(str(pg_path))
            w, h = img.size
            pg_num = int(pg_path.stem.split("-")[-1])
            meta = {
                "paper_id": paper_id,
                "pdf": pdf_path.name,
                "page": pg_num,
                "filename": pg_path.name,
                "path": str(pg_path),
                "width_px": w,
                "height_px": h,
                "type": "full_page",
                "extraction_status": "pending",
            }
            page_metas.append(meta)
            logger.info(f"  📄 Página {pg_num}: {w}×{h}px")
        except Exception as e:
            logger.warning(f"  ⚠️ {pg_path.name}: {e}")

    logger.info(f"✅ {paper_id}: {len(page_metas)} páginas renderizadas")
    return page_metas


def run(pdf_dir: str, output_dir: str, paper_filter: str = None) -> None:
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if paper_filter:
        pdfs = [p for p in pdfs if get_paper_id(p.name) == paper_filter.upper()]

    all_pages = []
    for pdf_path in pdfs:
        paper_id = get_paper_id(pdf_path.name)
        logger.info(f"\n📂 Renderizando {pdf_path.name} → {paper_id}")
        pages = render_pdf_pages(pdf_path, output_dir, paper_id)
        all_pages.extend(pages)

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)

    logger.info(f"\n✅ {len(all_pages)} páginas de {len(pdfs)} PDFs")
    logger.info(f"📁 Metadata: {metadata_path}")
    logger.info(f"\n💡 NOTA: Para extraer solo las páginas con figuras,")
    logger.info(f"   ejecuta extract_nps_data.py con --figures-dir {output_dir}")
    logger.info(f"   Claude Vision detectará si la página tiene datos extraíbles (extractable=true/false)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", default=os.getenv("PDF_SOURCE_DIR", "./pdfs"))
    parser.add_argument("--output-dir", default="./pages")
    parser.add_argument("--paper", default=None)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()
    run(args.pdf_dir, args.output_dir, args.paper)
