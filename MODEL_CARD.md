# Model card — PrintGuard AI

## Model details

| | Model A — detector | Model A — classifier | Model B — temporal |
|---|---|---|---|
| Task | box detection of `stringing` | frame classification: `extrusion_error`, `layer_shift`, `no_error` | per-time-step multi-label: `stringing`, `layer_shift` |
| Architecture | Ultralytics YOLO (nano) | Ultralytics YOLO-cls (nano) | causal TCN (4 levels, 64 ch) or small Transformer (~76 k params) |
| Input | RGB frame (640 px full, 256 px smoke) | RGB frame (224 px full, 128 px smoke) | window of 32 per-frame feature vectors |
| Training data | YOLO-Strings | Trier FFF (Bast et al. 2026) | YOLO-Strings sequences + Trier sequences with synthetic layer shift |
| Runtime | ONNX Runtime / PyTorch | ONNX Runtime / PyTorch | ONNX Runtime / PyTorch (server picks by file extension) |
| Author | Bilel Dhahri | | |
| License | MIT (code); datasets CC-BY-4.0 | | |

Model B's per-frame feature vector concatenates: a 128-d embedding from a frozen ImageNet
MobileNetV3-Small (pooled 576-d, fixed random projection), detector summary statistics per class, and
classifier probabilities. Model B never sees raw pixels, which keeps it small and fast.

## Intended use

- Decision support for operators of FFF 3D printers: flag likely stringing, extrusion errors and
  layer shifts, show the trend, and let a person decide whether to pause.
- Research / portfolio comparison of frame-level vs temporal defect detection.

## Out-of-scope use

- Autonomous control of a printer, or any setting where a missed alert could cause harm without a
  human checking. The system is deliberately built so pausing needs operator approval.
- Fire / thermal-runaway safety. This is not a safety device.
- Defects outside the taxonomy (spaghetti, warping, under-extrusion at nozzle level, etc.).
- Other manufacturing processes (SLA, SLS, metal AM) without retraining and new evaluation.

## Metrics

See the README "Evaluation methodology" section. Frame-level: precision, recall, mAP, accuracy,
macro-F1. Sequence-level (what matters on a printer): sequences detected, median detection delay,
false alarms per 100 clean frames, with a shared 3-of-5 alarm rule for both models.

## Current results — CPU smoke test, not final

| Model | Setting | Result |
|---|---|---|
| Detector | 1 epoch, CPU | P 0.060 · R 0.308 · mAP50 0.046 · mAP50-95 0.009 |
| Classifier | 3 epochs, CPU | accuracy 0.532 · macro-F1 0.517 |
| TCN vs A, stringing | test sequences | frame AP 0.965 vs 0.385 (likely inflated, see caveats) |
| TCN vs A, synthetic layer shift | test sequences | frame AP 0.485 vs 0.334 · detected 4/8 vs 4/8 · false alarms 1.08 vs 1.88 per 100 frames |
| Full pipeline | 2-core CPU, ONNX Model A + embedder, PyTorch TCN | ~13.8 FPS |
| Model B step | 2-core CPU, window 32, ONNX Runtime vs PyTorch | TCN 0.43 vs 5.16 ms · Transformer 0.39 vs 5.61 ms |

These are from minutes of CPU training on small subsets and only show that the pipeline and the
evaluation work end to end. They must not be quoted as model performance.

## Caveats and known biases

- **Scene shortcut (stringing, Model B).** Stringing and non-stringing sequences come from different
  prints, so temporal context can identify the print rather than the defect. The 0.965 AP is likely an
  over-estimate.
- **Synthetic layer shift.** Model B's layer-shift skill is measured on synthetic shifts; transfer to
  real gradual shifts is unverified.
- **Camera / printer domain.** Each dataset has its own printers, filament colours, lighting and camera
  angles. Expect degraded performance elsewhere until fine-tuned on local footage.
- **Assumed timing.** Per-print-hour rates assume a fixed time per frame because layer times are not
  published.
- **Class coverage.** No training data for spaghetti or warping.

## Ethical considerations

Training data contains printed parts only. Deployed cameras may incidentally capture people; crop to
the build plate and keep data local. Alerts can be wrong in both directions — keep a human in the loop.
