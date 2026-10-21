# PrintGuard AI

Real-time visual quality inspection for fused-filament (FFF) 3D printing.
A frame-level defect detector (Model A, YOLO) is compared against a temporal model
(Model B, TCN/Transformer over consecutive frames) that catches *evolving* failures
such as gradual layer shifts earlier. A FastAPI backend streams results to a live
quality-control dashboard. Printer pause is only ever requested through
operator-approved safety logic.

## Pipeline
1. Live camera + optional printer telemetry (layer index, status, temperatures)
2. Visual defect model (YOLO detection / segmentation)
3. Temporal analysis (TCN or Transformer over frame embeddings + detections)
4. QC decision support (defect type, region, confidence, trend, operator alert)

## Data
- Bast et al., *Quality Control in Additive Manufacturing: Image Dataset for Defect Detection in FFF*, Zenodo v2.0.0 (2026), https://doi.org/10.5281/zenodo.18628777 — extrusion errors, layer shifts, stringing, defect-free; 3 camera angles.
- YOLO-Strings (Zenodo 14711320) — stringing sequences, useful for temporal training.
- Own recordings once a printer + camera are available.

## Evaluation
Per-defect precision/recall, mAP@0.5 / mAP@0.5:0.95 (or mask IoU), false alarms per print-hour,
detection delay (frames/seconds from defect onset to alert), inference FPS (desktop + Raspberry Pi).

## Status
See [ROADMAP.md](ROADMAP.md).
