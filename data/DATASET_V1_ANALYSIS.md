# Dataset v1 — Análisis de Gaps (3-Mar-2026)

## Resumen ejecutivo

**🔴 0/50 filas están ML-ready** (todas las columnas target llenas).

La extracción de texto (Fase 1) capturó metadata (paper_id, metal_system, cell_line) pero los datos cuantitativos críticos están en **figuras** (gráficas de barras, curvas dosis-respuesta, DLS).

## Completitud por columna objetivo

| Columna | Descripción | Filled | % | Fuente probable |
|---------|-------------|--------|---|-----------------|
| `pdi` | Polydispersity Index | 0/50 | 0% | DLS figures, tables |
| `ic50_ug_ml` | IC50 (µg/mL) | 13/50 | 26% | Dose-response curves, text |
| `viability_pct` | Cell viability (%) | 0/50 | 0% | Bar charts (MTT/MTS) |
| `ros_pct` | ROS vs control (%) | 0/50 | 0% | Bar charts, fluorescence |
| `mic_ug_ml` | MIC (µg/mL) | 5/50 | 10% | Tables, text |
| `lspr_nm` | LSPR peak (nm) | — | — | UV-Vis spectra (column missing!) |

## Gaps por paper

| Paper | Rows | IC50 | Viab | ROS | PDI | MIC |
|-------|------|------|------|-----|-----|-----|
| P01 (AgAu starch 2019) | 4 | ✅ 4 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 |
| P02 (AuAg cardiac 2025) | 6 | ⚠️ 2 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 |
| P03 (AgAu MCF-7 2024) | 6 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 |
| P04 (SeTe pepper 2022) | 12 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 |
| P05 (AgAu melanoma 2022) | 8 | ✅ 4 | ❌ 0 | ❌ 0 | ❌ 0 | ❌ 0 |
| P07 (SeNPs 2023) | 14 | ⚠️ 3 | ❌ 0 | ❌ 0 | ❌ 0 | — |

## Impacto esperado del Pipeline (Fase 2-4)

Si la extracción de figuras funciona bien, podríamos llenar:
- **PDI:** ~30-40 valores (de gráficas DLS en todos los papers)
- **Viabilidad:** ~20-30 valores (de bar charts MTT/MTS en P01-P05, P07)
- **ROS:** ~10-15 valores (solo papers con ensayos ROS)
- **IC50:** +10-15 valores adicionales (de curvas dosis-respuesta)
- **LSPR:** Columna nueva con ~20-30 valores (de espectros UV-Vis)

**Estimación conservadora:** pasar de 0% ML-ready → 40-60% ML-ready.

## Acción requerida

**Jorge necesita subir los PDFs P01-P07** a:
```
/mnt/agent-workspace/Investigación/Cholula-Papers/
```
Nombrados: `01_AgAu_starch_2019.pdf`, `02_AuAg_cardiac_2025.pdf`, etc.

Pipeline listo para ejecutar: `./run_pipeline.sh`
