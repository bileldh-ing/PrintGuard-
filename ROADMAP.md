# Roadmap

Status legend: `[x]` done and in this repo · `[~]` partly done (see note) · `[ ]` not started.

Compute note: development happened on a 2-core CPU sandbox with no GPU. Everything below marked done
has been run there at **smoke scale**. Full-scale training runs from the Colab GPU notebooks in
`notebooks/`.

## Phase 1 — Foundations
- [x] 1. Repo structure (`printguard/`, `server/`, `configs/`, `notebooks/`, `tests/`), pyproject, MIT license, ruff config
- [x] 2. Dataset downloader for Zenodo 18628777 (Trier FFF) + 14711320 (YOLO-Strings): ranged member reads, CRC-32 / MD5 checks, data card
- [~] 3. EDA notebook (`notebooks/01_eda.ipynb`) — written; to be executed with outputs on the full dataset
- [x] 4. Labels converted to YOLO format; train/val/test split by recording sequence / print block, with a leakage check that fails the build

## Phase 2 — Model A (frame level)
- [x] 5. YOLO detector + YOLO classifier training configs (full + smoke) and Colab notebook (`02_train_model_a_colab.ipynb`)
- [x] 6. Evaluation script: per-class P/R, mAP@0.5 / 0.5:0.95, accuracy, macro-F1, confusion matrix
- [ ] 7. Error-analysis notebook (false positive / false negative gallery)
- [x] 8. ONNX export (detector, classifier, embedder) and CPU FPS benchmark (~13.8 FPS end to end on 2 cores)

## Phase 3 — Model B (temporal)
- [x] 9. Sequence dataset builder: resumable per-sequence feature cache (Model A outputs + frozen MobileNetV3-S embedding), sliding windows
- [x] 10. Synthetic gradual layer-shift augmentation for sequences
- [x] 11. Causal TCN baseline + masked multi-label training loop
- [x] 12. Small Transformer variant
- [x] 13. Detection delay + false alarms per 100 frames / per print-hour; A vs B comparison report (smoke scale)

## Phase 4 — System
- [x] 14. Video / camera ingestion (OpenCV: USB, RTSP, file replay with rate limiting)
- [x] 15. FastAPI backend: inference worker, WebSocket events, SQLite timeline (shutdown-safe)
- [x] 16. Live dashboard — implemented as a dependency-free vanilla-JS single page instead of React
- [x] 17. Operator alert + pause-request safety logic (OctoPrint / Moonraker, approval required, disabled + dry-run by default, audited)
- [~] 18. Dockerfile + docker compose done; Raspberry Pi guide written (`docs/raspberry_pi.md`) but not yet measured on a real Pi

## Phase 5 — Proof
- [~] 19. Reproducible eval pipeline (`make eval`) and results table in README — **smoke-test numbers only**; full GPU numbers pending
- [~] 20. Demo GIF and architecture diagram done; write-up / blog post and demo video with narration pending

## Next (after the core roadmap)
- [ ] Full GPU training of Model A and Model B, update README results
- [ ] Scene-matched negatives for stringing sequences (remove the scene shortcut in Model B's stringing AP)
- [ ] Measure FPS on Raspberry Pi 5 / Jetson
- [ ] Own recordings with a printer + camera; add `spaghetti` and `warping` classes
- [ ] Use printer telemetry (layer index from OctoPrint / Moonraker) to align frames with layers
