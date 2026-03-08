#!/usr/bin/env python3
"""
Fase 3 Piloto v2: Extracción de datos cuantitativos de P01 usando PÁGINAS COMPLETAS.

Reason: pdfimages strips axis labels from charts (separate text layers).
Full page renders (pdftoppm) preserve all axis labels, legends, and data.

Paper P01 = "AgAu bimetallic NPs synthesized with starch" (antimicrobial, 2019)
"""

import anthropic
import base64
import json
import time
import sys
from pathlib import Path
from datetime import datetime

# Config
API_KEY = Path("/home/jorge/.openclaw/secrets/anthropic_api_key.txt").read_text().strip()
PAGES_DIR = Path("/home/jorge/workspace/plot-digitizer-nps/pages/P01")
OUTPUT_PATH = Path("/home/jorge/workspace/plot-digitizer-nps/extraction_results/P01_results.json")

client = anthropic.Anthropic(api_key=API_KEY)

PROMPT = """You are a scientific data extraction specialist. You are analyzing a PAGE from a research paper about bimetallic AgAu nanoparticles synthesized with starch as a green stabilizer (2019 paper on antimicrobial and cytotoxicity applications).

CAREFULLY examine this page. If it contains ANY figures, charts, tables, or data:

**Extract ALL quantitative values you can read from the figures/tables on this page.**

Target parameters (extract any you find):
1. **size_nm**: Nanoparticle diameter in nm (from TEM histograms, DLS data, or text)
2. **pdi**: Polydispersity index (0-1) from DLS measurements
3. **zeta_mv**: Zeta potential in mV
4. **ag_at_pct / au_at_pct**: Atomic % of Ag and Au (from EDX spectra/tables)
5. **viabilidad_pct**: Cell viability % (from MTT/MTS cytotoxicity bar charts or dose-response curves)
6. **ic50_ug_ml**: IC50 in μg/mL (concentration causing 50% cell death)
7. **mic_ug_ml**: MIC in μg/mL (minimum inhibitory concentration, antimicrobial)
8. **ros_pct**: ROS % relative to control (reactive oxygen species)
9. **lspr_nm**: UV-Vis LSPR peak wavelength in nm
10. **zone_inhibition_mm**: Zone of inhibition in mm (antimicrobial disk diffusion)

READ THE AXES CAREFULLY:
- X-axis labels and values
- Y-axis labels and values  
- Data labels on bars/points
- Legend entries matching data series to NP compositions
- Error bars (report as error_margin)
- Table values (read each cell)

For NP compositions, identify: Ag, Au, Ag20Au80, Ag40Au60, Ag50Au50, Ag60Au40, Ag80Au20, etc.
For cell lines: MCF-7, HeLa, H9c2, HDF, etc.
For bacteria: E. coli, S. aureus, MRSA, B. subtilis, P. aeruginosa, etc.

RESPOND ONLY IN VALID JSON (no markdown, no code fences):
{
  "page_has_figures": true/false,
  "page_has_tables": true/false,
  "page_description": "brief description of what's on this page",
  "figures_found": [
    {
      "figure_id": "Fig. 1a or similar",
      "figure_type": "bar_chart|line_plot|dose_response|histogram|size_distribution|UV-Vis_spectrum|EDX_spectrum|XRD|FTIR|TEM|SEM|DLS|zeta_potential|table|schematic|zone_inhibition|other",
      "x_axis_label": "the x-axis label",
      "y_axis_label": "the y-axis label",
      "data_points": [
        {
          "parameter": "size_nm|pdi|zeta_mv|ag_at_pct|au_at_pct|viabilidad_pct|ic50_ug_ml|mic_ug_ml|ros_pct|lspr_nm|zone_inhibition_mm|absorbance|other",
          "value": 123.4,
          "error_margin": 5.2,
          "condition": "Ag50Au50 NPs, 100 μg/mL, E. coli, 24h",
          "units": "nm|mV|%|μg/mL|mm|a.u.",
          "confidence": "high|medium|low"
        }
      ]
    }
  ],
  "tables_found": [
    {
      "table_id": "Table 1 or similar",
      "table_description": "what the table contains",
      "rows": [
        {"column1": "value1", "column2": "value2"}
      ]
    }
  ],
  "text_data_mentioned": [
    {
      "parameter": "size_nm|pdi|zeta_mv|etc",
      "value": 45.2,
      "condition": "from text description",
      "confidence": "high"
    }
  ]
}

If the page is only text with no figures/tables/quantitative data, respond:
{
  "page_has_figures": false,
  "page_has_tables": false,
  "page_description": "text only - abstract/introduction/methods/etc",
  "figures_found": [],
  "tables_found": [],
  "text_data_mentioned": []
}"""


def analyze_page(img_path: Path, page_num: int) -> tuple:
    """Analyze a full page with Claude Vision."""
    img_data = base64.standard_b64encode(img_path.read_bytes()).decode('utf-8')
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_data
                    }
                },
                {"type": "text", "text": PROMPT}
            ]
        }]
    )
    
    text = response.content[0].text.strip()
    tokens = response.usage.input_tokens + response.usage.output_tokens
    
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                result = {"error": "JSON parse failed", "raw": text[:500]}
        else:
            result = {"error": "No JSON found", "raw": text[:500]}
    
    return result, tokens


def main():
    # Get all page images (skip tiny ones like page 14 = 8KB likely blank)
    pages = sorted(PAGES_DIR.glob("P01-*.png"))
    pages = [p for p in pages if p.stat().st_size > 50000]  # Skip tiny/blank pages
    
    print(f"🔬 Cholula ML — Fase 3 Piloto v2: Extracción por PÁGINAS COMPLETAS")
    print(f"📄 Paper: P01 (AgAu starch antimicrobial 2019)")
    print(f"📑 Páginas: {len(pages)}")
    print(f"🤖 Modelo: claude-opus-4-6")
    print(f"⏰ Inicio: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    results = []
    total_tokens = 0
    pages_with_figures = 0
    all_data_points = []
    
    for i, page_path in enumerate(pages):
        page_num = int(page_path.stem.split('-')[1])
        print(f"[{i+1}/{len(pages)}] Página {page_num} ({page_path.name}, {page_path.stat().st_size//1024}KB)...", end=" ", flush=True)
        
        try:
            extracted, tokens = analyze_page(page_path, page_num)
            total_tokens += tokens
            
            has_figs = extracted.get('page_has_figures', False)
            has_tables = extracted.get('page_has_tables', False)
            n_figs = len(extracted.get('figures_found', []))
            n_tables = len(extracted.get('tables_found', []))
            n_text_data = len(extracted.get('text_data_mentioned', []))
            desc = extracted.get('page_description', '')[:60]
            
            # Count data points
            page_dp = 0
            for fig in extracted.get('figures_found', []):
                page_dp += len(fig.get('data_points', []))
                all_data_points.extend(fig.get('data_points', []))
            for tbl in extracted.get('tables_found', []):
                page_dp += len(tbl.get('rows', []))
            all_data_points.extend(extracted.get('text_data_mentioned', []))
            page_dp += n_text_data
            
            if has_figs or has_tables:
                pages_with_figures += 1
                print(f"✅ {n_figs} figs, {n_tables} tables, {page_dp} datapoints ({tokens} tok) — {desc}")
            else:
                print(f"⬜ text only ({tokens} tok) — {desc}")
            
            results.append({
                "page_num": page_num,
                "filename": page_path.name,
                "file_size_bytes": page_path.stat().st_size,
                "status": "done",
                "extraction": extracted,
                "tokens_used": tokens
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "page_num": page_num,
                "filename": page_path.name,
                "status": "error",
                "error": str(e)
            })
        
        # Rate limiting
        if i < len(pages) - 1:
            time.sleep(3)
    
    # Build comprehensive output
    output = {
        "meta": {
            "paper_id": "P01",
            "paper_title": "AgAu bimetallic NPs synthesized with starch - antimicrobial (2019)",
            "extraction_method": "full_page_claude_vision",
            "extraction_model": "claude-opus-4-6",
            "extraction_date": datetime.now().isoformat(),
            "total_pages_analyzed": len(pages),
            "pages_with_figures_or_tables": pages_with_figures,
            "total_data_points_extracted": len(all_data_points),
            "total_tokens_used": total_tokens,
            "estimated_cost_usd": round(total_tokens * 0.00003, 4)
        },
        "pages": results,
        "summary": {
            "parameters_found": {},
            "all_extracted_data": []
        }
    }
    
    # Collect all data points with page context
    for r in results:
        if r["status"] != "done":
            continue
        ext = r["extraction"]
        page_num = r["page_num"]
        
        for fig in ext.get("figures_found", []):
            fig_id = fig.get("figure_id", "unknown")
            fig_type = fig.get("figure_type", "unknown")
            for dp in fig.get("data_points", []):
                dp_with_context = {
                    **dp,
                    "source_page": page_num,
                    "source_figure": fig_id,
                    "source_figure_type": fig_type
                }
                output["summary"]["all_extracted_data"].append(dp_with_context)
                param = dp.get("parameter", "other")
                output["summary"]["parameters_found"][param] = output["summary"]["parameters_found"].get(param, 0) + 1
        
        for tbl in ext.get("tables_found", []):
            tbl_id = tbl.get("table_id", "unknown")
            for row in tbl.get("rows", []):
                output["summary"]["all_extracted_data"].append({
                    "source_page": page_num,
                    "source_table": tbl_id,
                    "row_data": row
                })
        
        for td in ext.get("text_data_mentioned", []):
            td_with_context = {**td, "source_page": page_num, "source": "text"}
            output["summary"]["all_extracted_data"].append(td_with_context)
            param = td.get("parameter", "other")
            output["summary"]["parameters_found"][param] = output["summary"]["parameters_found"].get(param, 0) + 1
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 RESUMEN FINAL — P01 Extracción por Páginas")
    print(f"{'='*70}")
    print(f"  Páginas analizadas:        {len(pages)}")
    print(f"  Con figuras/tablas:        {pages_with_figures}")
    print(f"  Total data points:         {len(output['summary']['all_extracted_data'])}")
    print(f"  Tokens usados:             {total_tokens:,}")
    print(f"  Costo estimado:            ${total_tokens * 0.00003:.4f} USD")
    
    print(f"\n📈 Parámetros encontrados:")
    for param, count in sorted(output["summary"]["parameters_found"].items(), key=lambda x: -x[1]):
        print(f"    {param}: {count} valores")
    
    # Print all key extracted values grouped by parameter
    print(f"\n{'='*70}")
    print(f"🔑 TODOS LOS VALORES EXTRAÍDOS (agrupados por parámetro)")
    print(f"{'='*70}")
    
    target_params = ['size_nm', 'pdi', 'zeta_mv', 'ag_at_pct', 'au_at_pct', 
                     'viabilidad_pct', 'ic50_ug_ml', 'mic_ug_ml', 'ros_pct', 
                     'lspr_nm', 'zone_inhibition_mm']
    
    for param in target_params:
        matching = [dp for dp in output["summary"]["all_extracted_data"] 
                   if dp.get("parameter") == param]
        if matching:
            print(f"\n  📊 {param}:")
            for dp in matching:
                val = dp.get("value", "?")
                err = f" ± {dp['error_margin']}" if dp.get("error_margin") else ""
                units = dp.get("units", "")
                cond = dp.get("condition", "")
                conf = dp.get("confidence", "?")
                page = dp.get("source_page", "?")
                fig = dp.get("source_figure", dp.get("source", "?"))
                print(f"    • {val}{err} {units} | {cond} | pg.{page}/{fig} (conf: {conf})")
    
    # Also print "other" params
    other_vals = [dp for dp in output["summary"]["all_extracted_data"] 
                 if dp.get("parameter") not in target_params and "parameter" in dp]
    if other_vals:
        print(f"\n  📊 Otros parámetros:")
        for dp in other_vals[:20]:  # limit to 20
            param = dp.get("parameter", "?")
            val = dp.get("value", "?")
            cond = dp.get("condition", "")
            page = dp.get("source_page", "?")
            print(f"    • {param} = {val} | {cond} | pg.{page}")
    
    # Print table data
    table_data = [dp for dp in output["summary"]["all_extracted_data"] if "row_data" in dp]
    if table_data:
        print(f"\n  📋 Datos de tablas ({len(table_data)} rows):")
        for td in table_data[:10]:
            print(f"    pg.{td.get('source_page', '?')}/{td.get('source_table', '?')}: {json.dumps(td['row_data'], ensure_ascii=False)[:120]}")
    
    print(f"\n💾 Guardado en: {OUTPUT_PATH}")
    print(f"⏰ Fin: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
