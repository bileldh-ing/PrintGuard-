# Roadmap (one item per daily session)

Compute note: development sandbox is CPU-only. Heavy training runs as Colab/Kaggle GPU notebooks in `notebooks/`; CPU smoke tests run in CI-sized subsets.

## Phase 1 — Foundations
- [ ] 1. Repo structure (`printguard/`, `configs/`, `notebooks/`, `tests/`), pyproject, license, pre-commit
- [ ] 2. Dataset downloader for Zenodo 18628777 + YOLO-Strings, checksum, data card
- [ ] 3. EDA notebook: class balance, camera angles, image sizes, sample grid
- [ ] 4. Convert labels to YOLO format; train/val/test split by *print* (no leakage)

## Phase 2 — Model A (frame detector)
- [ ] 5. YOLOv8n/YOLO11n training config + Colab notebook
- [ ] 6. Evaluation script: per-class P/R, mAP, confusion matrix, PR curves
- [ ] 7. Error analysis notebook (false positives/negatives gallery)
- [ ] 8. Export to ONNX; CPU/RPi FPS benchmark script

## Phase 3 — Model B (temporal)
- [ ] 9. Sequence dataset builder (sliding windows of frame embeddings + detections)
- [ ] 10. Synthetic gradual layer-shift augmentation for sequences
- [ ] 11. TCN baseline model + training loop
- [ ] 12. Small Transformer variant
- [ ] 13. Detection-delay + false-alarms-per-print-hour metrics; A vs B comparison report

## Phase 4 — System
- [ ] 14. Video/camera ingestion service (OpenCV, RTSP/USB, file replay)
- [ ] 15. FastAPI backend: inference worker, WebSocket events, SQLite timeline
- [ ] 16. Dashboard (React): live video, boxes, confidence, issue timeline
- [ ] 17. Operator alert + pause-request safety logic (OctoPrint/Moonraker, approval required)
- [ ] 18. Docker compose; Raspberry Pi deployment guide

## Phase 5 — Proof
- [ ] 19. Reproducible eval pipeline (`make eval`), results tables in README
- [ ] 20. Demo video script, architecture diagram, write-up / blog post, CV bullet
