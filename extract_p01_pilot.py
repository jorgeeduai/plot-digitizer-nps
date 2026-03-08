#!/usr/bin/env python3
"""
Fase 3 Piloto: Extracción de datos cuantitativos de las 15 figuras candidatas de P01
usando Claude Vision (claude-opus-4-6).

Paper P01 = AgAu bimetallic NPs synthesized with starch (antimicrobial, 2019).
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
FIGURES_DIR = Path("/home/jorge/workspace/plot-digitizer-nps/figures/P01")
CANDIDATES_JSON = Path("/home/jorge/workspace/plot-digitizer-nps/figures/candidates.json")
OUTPUT_PATH = Path("/home/jorge/workspace/plot-digitizer-nps/extraction_results/P01_results.json")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=API_KEY)

# Specialized prompt for P01 (AgAu starch antimicrobial NPs)
PROMPT = """You are analyzing a figure from a scientific paper about bimetallic AgAu nanoparticles synthesized with starch as stabilizer (2019 paper on antimicrobial applications).

Extract ALL quantitative data visible in this figure. Look for:

1. **Nanoparticle size (nm)**: from TEM images (scale bars, particle measurements), DLS charts, or size distribution histograms
2. **PDI (polydispersity index, 0-1)**: from DLS data
3. **Zeta potential (mV)**: from zeta potential measurements
4. **Atomic composition (% Ag, % Au)**: from EDX spectra, tables, or elemental mapping
5. **Cell viability (%)**: from MTT/MTS cytotoxicity assays — bar charts showing % viability at different concentrations
6. **IC50 (μg/mL)**: from dose-response curves — the concentration causing 50% inhibition
7. **MIC (μg/mL)**: from antimicrobial assays — minimum inhibitory concentration
8. **ROS (%)**: from oxidative stress assays — reactive oxygen species relative to control
9. **UV-Vis/LSPR peak (nm)**: surface plasmon resonance peak wavelength

For each value found, note:
- The exact numeric value(s) — read carefully from axes/bars/data labels
- The units
- What experimental condition it corresponds to (e.g., "Ag20Au80 NPs", "100 μg/mL", "E. coli", "24h exposure")
- Your confidence: high (clearly readable), medium (estimated from chart position), low (uncertain)

IMPORTANT RULES:
- If this is a TEM/SEM/AFM microscopy image: look for scale bars and estimate particle sizes if possible. If no quantitative data at all, set contains_target_data=false.
- If this is an EDX spectrum: extract elemental percentages (Ag, Au, and any others visible).
- If this is a UV-Vis spectrum: extract LSPR peak position(s).
- If this is a bar chart or line plot: extract ALL data points visible.
- For dose-response curves: extract viability at each concentration tested.

Respond ONLY in valid JSON format (no markdown, no code blocks):
{
  "figure_type": "bar_chart|line_plot|scatter_plot|histogram|size_distribution|dose_response|TEM|SEM|EDX_spectrum|UV-Vis|XRD|FTIR|zeta_potential|table|schematic|other",
  "contains_target_data": true or false,
  "extracted_values": [
    {
      "parameter": "size_nm|pdi|zeta_mv|ag_at_pct|au_at_pct|viabilidad_pct|ic50_ug_ml|mic_ug_ml|ros_pct|lspr_nm|other",
      "value": 123.4,
      "error_margin": 5.2,
      "value_range": [100, 150],
      "condition": "description of experimental condition",
      "units": "nm|mV|%|μg/mL",
      "confidence": "high|medium|low"
    }
  ],
  "notes": "any important observations about this figure"
}

If no quantitative target data is extractable, use:
{
  "figure_type": "...",
  "contains_target_data": false,
  "extracted_values": [],
  "notes": "reason why no target data found"
}"""


def analyze_figure(img_path: Path) -> dict:
    """Analyze a single figure with Claude Vision."""
    img_data = base64.standard_b64encode(img_path.read_bytes()).decode('utf-8')
    
    # Determine media type
    suffix = img_path.suffix.lower()
    media_types = {'.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg'}
    media_type = media_types.get(suffix, 'image/png')
    
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data
                    }
                },
                {"type": "text", "text": PROMPT}
            ]
        }]
    )
    
    text = response.content[0].text.strip()
    tokens = response.usage.input_tokens + response.usage.output_tokens
    
    # Parse JSON from response
    try:
        # Try direct parse first
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON block in response
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                result = {"error": "JSON parse failed", "raw_response": text[:500]}
        else:
            result = {"error": "No JSON found", "raw_response": text[:500]}
    
    return result, tokens


def main():
    # Load candidates for P01
    candidates = json.loads(CANDIDATES_JSON.read_text())
    p01_candidates = [c for c in candidates if c['paper_id'] == 'P01']
    
    print(f"🔬 Cholula ML — Fase 3 Piloto: Extracción con Claude Vision")
    print(f"📄 Paper: P01 (AgAu starch antimicrobial 2019)")
    print(f"🖼️  Candidatas: {len(p01_candidates)} figuras")
    print(f"🤖 Modelo: claude-opus-4-6")
    print(f"⏰ Inicio: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    results = []
    total_tokens = 0
    figures_with_data = 0
    all_extracted_values = []
    
    for i, candidate in enumerate(p01_candidates):
        img_path = Path("/home/jorge/workspace/plot-digitizer-nps") / candidate['path']
        filename = candidate['filename']
        
        print(f"[{i+1}/{len(p01_candidates)}] Procesando {filename}...", end=" ", flush=True)
        
        if not img_path.exists():
            print(f"❌ ARCHIVO NO ENCONTRADO: {img_path}")
            results.append({
                "filename": filename,
                "path": candidate['path'],
                "paper_id": "P01",
                "status": "error",
                "error": "File not found"
            })
            continue
        
        try:
            extracted, tokens = analyze_figure(img_path)
            total_tokens += tokens
            
            has_data = extracted.get('contains_target_data', False)
            n_values = len(extracted.get('extracted_values', []))
            fig_type = extracted.get('figure_type', 'unknown')
            
            if has_data:
                figures_with_data += 1
                all_extracted_values.extend(extracted.get('extracted_values', []))
                print(f"✅ {fig_type} — {n_values} valores ({tokens} tokens)")
            else:
                print(f"⬜ {fig_type} — sin datos target ({tokens} tokens)")
            
            results.append({
                "filename": filename,
                "path": candidate['path'],
                "paper_id": "P01",
                "file_size_bytes": img_path.stat().st_size,
                "status": "done",
                "extraction": extracted
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "filename": filename,
                "path": candidate['path'],
                "paper_id": "P01",
                "status": "error",
                "error": str(e)
            })
        
        # Rate limiting — be nice to the API
        if i < len(p01_candidates) - 1:
            time.sleep(2)
    
    # Save results
    output = {
        "meta": {
            "paper_id": "P01",
            "paper_title": "AgAu bimetallic NPs synthesized with starch - antimicrobial (2019)",
            "extraction_model": "claude-opus-4-6",
            "extraction_date": datetime.now().isoformat(),
            "total_candidates": len(p01_candidates),
            "figures_with_data": figures_with_data,
            "figures_without_data": len(p01_candidates) - figures_with_data,
            "total_extracted_values": len(all_extracted_values),
            "total_tokens_used": total_tokens,
            "estimated_cost_usd": round(total_tokens * 0.00003, 4)  # rough estimate
        },
        "results": results,
        "summary": {
            "parameters_found": {},
            "figure_types": {}
        }
    }
    
    # Build summary
    for val in all_extracted_values:
        param = val.get('parameter', 'unknown')
        output["summary"]["parameters_found"][param] = output["summary"]["parameters_found"].get(param, 0) + 1
    
    for r in results:
        if r["status"] == "done":
            ft = r["extraction"].get("figure_type", "unknown")
            output["summary"]["figure_types"][ft] = output["summary"]["figure_types"].get(ft, 0) + 1
    
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Total figuras procesadas: {len(results)}")
    print(f"  Con datos extraíbles:     {figures_with_data}")
    print(f"  Sin datos target:         {len(results) - figures_with_data}")
    print(f"  Total valores extraídos:  {len(all_extracted_values)}")
    print(f"  Tokens usados:            {total_tokens:,}")
    print(f"  Costo estimado:           ${total_tokens * 0.00003:.4f} USD")
    print(f"\n📈 Parámetros encontrados:")
    for param, count in sorted(output["summary"]["parameters_found"].items(), key=lambda x: -x[1]):
        print(f"    {param}: {count} valores")
    print(f"\n🖼️  Tipos de figura:")
    for ft, count in sorted(output["summary"]["figure_types"].items(), key=lambda x: -x[1]):
        print(f"    {ft}: {count}")
    
    print(f"\n💾 Resultados guardados en: {OUTPUT_PATH}")
    print(f"⏰ Fin: {datetime.now().isoformat()}")
    
    # Print key extracted values for report
    print(f"\n{'='*60}")
    print(f"🔑 VALORES CLAVE EXTRAÍDOS")
    print(f"{'='*60}")
    for r in results:
        if r["status"] == "done" and r["extraction"].get("contains_target_data"):
            print(f"\n  📊 {r['filename']} ({r['extraction']['figure_type']}):")
            for val in r["extraction"].get("extracted_values", []):
                param = val.get("parameter", "?")
                value = val.get("value", "?")
                units = val.get("units", "")
                condition = val.get("condition", "")
                conf = val.get("confidence", "?")
                err = f" ± {val['error_margin']}" if val.get("error_margin") else ""
                print(f"    • {param} = {value}{err} {units} [{condition}] (conf: {conf})")


if __name__ == "__main__":
    main()
