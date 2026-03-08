#!/usr/bin/env python3
"""
Fase 3 Piloto: Extract data from P01 figures using Claude Vision API.
Only processes figures classified as data charts (not microscopy/logos).
"""

import anthropic
import base64
import json
import os
import time
import sys
from pathlib import Path

# === CONFIGURATION ===

API_KEY_PATH = "/home/jorge/.openclaw/secrets/anthropic_api_key.txt"
FIGURES_DIR = Path("/home/jorge/workspace/plot-digitizer-nps/figures/P01")
OUTPUT_DIR = FIGURES_DIR / "extractions"
MODEL = "claude-opus-4-6"
MAX_TOKENS = 2000
DELAY_SECONDS = 3  # rate limiting between API calls

# Paper context for P01
PAPER_CONTEXT = """This paper is: "Starch-stabilized bimetallic Ag-Au nanoparticles: 
synthesis, characterization, antibacterial and cytotoxic properties" (2019).
It studies AgAu bimetallic nanoparticles with different Ag:Au ratios 
(likely Ag20Au80, Ag40Au60, Ag50Au50, Ag60Au40, Ag80Au20, pure AgNPs, pure AuNPs).
Tests include: antibacterial activity (E. coli, S. aureus), cytotoxicity, 
DLS (size, PDI, zeta potential), UV-Vis (LSPR), XRD, FTIR, TEM."""

EXTRACTION_PROMPT = f"""You are a scientific data extraction specialist.

{PAPER_CONTEXT}

Analyze this figure and extract ALL numerical data visible.

**Target variables** (extract only those visible):
- PDI (polydispersity index): dimensionless value 0-1 from DLS measurements
- IC50: concentration causing 50% inhibition (in µg/mL or µM — note the units!)
- Cell viability (%): percentage of viable cells vs control (often at specific concentrations)
- ROS (%): reactive oxygen species relative to control (100% = same as control)
- MIC (µg/mL): minimum inhibitory concentration for antimicrobial data
- LSPR (nm): localized surface plasmon resonance peak wavelength
- Particle size (nm): from DLS or TEM
- Zeta potential (mV): surface charge

For growth curves: extract the OD values at key timepoints and identify which NP condition each curve represents.

**Output format** (strict JSON, no markdown):
{{
  "chart_type": "bar|scatter|line|histogram|spectrum|dose_response|growth_curve|dual_axis|other",
  "figure_description": "Brief description of what this figure shows",
  "x_axis": {{"label": "...", "unit": "..."}},
  "y_axis": {{"label": "...", "unit": "..."}},
  "data_points": [
    {{
      "condition": "description (e.g., Ag50Au50, AgNPs, 25 µg/mL)",
      "x_value": "<number or null>",
      "y_value": "<number or null>",
      "error_bar": "<number or null>",
      "variable_extracted": "pdi|ic50_ug_ml|ic50_um|viability_pct|ros_pct|mic_ug_ml|lspr_nm|size_nm|zeta_mv|od600|other",
      "confidence": "high|medium|low"
    }}
  ],
  "np_compositions": ["list of NP compositions shown"],
  "cell_lines": ["if applicable"],
  "bacteria": ["if applicable"],
  "concentrations_tested": ["if applicable"],
  "notes": "any important context",
  "extractable": true
}}

If this is NOT a data chart (microscopy, schematic, logo), respond:
{{"extractable": false, "chart_type": "microscopy|schematic|logo", "figure_description": "brief description"}}

Be PRECISE with numbers. Read values from axes carefully.
"""

# Figures classified as NOT data charts (skip these)
SKIP_FIGURES = {
    "P01-000.png",  # Facebook logo
    "P01-001.png",  # LinkedIn logo
    "P01-002.png",  # Twitter logo
    "P01-003.png",  # YouTube logo
    "P01-010.png",  # Photo of vials
    "P01-024.png",  # TEM nanoparticles
    "P01-025.png",  # TEM nanoparticles
    "P01-050.png",  # SEM bacteria (bacilli)
    "P01-051.png",  # SEM bacteria (bacilli)
    "P01-052.png",  # SEM bacteria (cocci)
    "P01-053.png",  # SEM bacteria (cocci)
    "P01-077.png",  # TEM dark-field
    "P01-092.png",  # SEM
    "P01-093.png",  # SEM
}

# Priority figures (most likely to contain target variables)
# Bar charts with viability/MIC/IC50, growth curves, size distributions
PRIORITY_FIGURES = [
    # Bar charts (likely viability, MIC, or IC50 data)
    "P01-054.png", "P01-055.png", "P01-056.png", "P01-057.png",
    "P01-046.png", "P01-047.png", "P01-048.png", "P01-049.png",
    "P01-020.png",
    # Growth curves (bacterial - MIC data)
    "P01-028.png", "P01-029.png", "P01-030.png", "P01-031.png",
    # Line graphs (dose-response, PDI, zeta potential)
    "P01-038.png", "P01-039.png", "P01-040.png", "P01-041.png",
    "P01-042.png", "P01-043.png",
    # Dual-axis plots (likely size + zeta or size + PDI)
    "P01-079.png", "P01-066.png",
    # Size distributions
    "P01-021.png", "P01-026.png", "P01-027.png",
    # Spectra (UV-Vis for LSPR)
    "P01-005.png", "P01-065.png", "P01-080.png",
    "P01-058.png", "P01-078.png",
    # DLS peaks
    "P01-011.png", "P01-012.png",
]


def extract_figure(client, fig_path: Path) -> dict:
    """Send a figure to Claude Vision and get structured data back."""
    with open(fig_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img_data,
                    },
                },
                {"type": "text", "text": EXTRACTION_PROMPT},
            ],
        }],
    )
    
    raw_text = response.content[0].text.strip()
    
    # Parse JSON
    try:
        extracted = json.loads(raw_text)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = {"error": "JSON parse failed", "raw": raw_text[:500]}
    
    return {
        "filename": fig_path.name,
        "extracted_data": extracted,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def main():
    # Load API key
    api_key = open(API_KEY_PATH).read().strip()
    client = anthropic.Anthropic(api_key=api_key)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get list of figures to process
    figures_to_process = PRIORITY_FIGURES
    
    print(f"\n🔬 Fase 3 Piloto — P01 Data Extraction")
    print(f"   Total figures in P01: 47")
    print(f"   Skipped (not data charts): {len(SKIP_FIGURES)}")
    print(f"   To process: {len(figures_to_process)}")
    print(f"   Model: {MODEL}")
    print()
    
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    for i, fig_name in enumerate(figures_to_process):
        fig_path = FIGURES_DIR / fig_name
        if not fig_path.exists():
            print(f"  ⚠️  [{i+1}/{len(figures_to_process)}] {fig_name} — NOT FOUND, skipping")
            continue
        
        # Check if already extracted
        output_file = OUTPUT_DIR / f"{fig_path.stem}_extraction.json"
        if output_file.exists():
            print(f"  ⏭️  [{i+1}/{len(figures_to_process)}] {fig_name} — already extracted, loading")
            with open(output_file) as f:
                result = json.load(f)
            all_results.append(result)
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
            continue
        
        print(f"  🔍 [{i+1}/{len(figures_to_process)}] {fig_name}...", end=" ", flush=True)
        
        try:
            result = extract_figure(client, fig_path)
            
            # Save individual result
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            all_results.append(result)
            total_input_tokens += result.get("input_tokens", 0)
            total_output_tokens += result.get("output_tokens", 0)
            
            extractable = result["extracted_data"].get("extractable", True)
            dp_count = len(result["extracted_data"].get("data_points", []))
            chart_type = result["extracted_data"].get("chart_type", "unknown")
            print(f"✅ {chart_type}, {dp_count} data points")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result = {"filename": fig_name, "error": str(e)}
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)
        
        # Rate limiting
        if i < len(figures_to_process) - 1:
            time.sleep(DELAY_SECONDS)
    
    # Save combined results
    combined_file = OUTPUT_DIR / "all_extractions.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n📊 SUMMARY")
    print(f"   Processed: {len(all_results)}")
    print(f"   Input tokens: {total_input_tokens:,}")
    print(f"   Output tokens: {total_output_tokens:,}")
    
    # Cost estimate (Claude Opus pricing)
    input_cost = total_input_tokens * 15 / 1_000_000  # $15/M input
    output_cost = total_output_tokens * 75 / 1_000_000  # $75/M output
    total_cost = input_cost + output_cost
    print(f"   Estimated cost: ${total_cost:.2f} (input: ${input_cost:.2f}, output: ${output_cost:.2f})")
    print(f"\n   Results saved to: {combined_file}")


if __name__ == "__main__":
    main()
