# Architecture

PrintGuard AI has two halves that share one library (`printguard/`):

1. an **offline pipeline** – download → prepare → train → export → evaluate – driven by `make`, and
2. an **online system** – ingestion → inference worker → alerts → API / dashboard → operator-approved pause.

## Offline pipeline

```mermaid
flowchart TD
    Z1[Zenodo 18628777<br/>Trier FFF] --> DL
    Z2[Zenodo 14711320<br/>YOLO-Strings] --> DL
    DL[data/download.py<br/>ranged member reads, CRC-32 / MD5] --> PREP
    PREP[data/prepare.py<br/>YOLO labels, group-wise splits,<br/>leakage check] --> CLS[(trier_cls/)]
    PREP --> YOLO[(yolo_strings/)]
    PREP --> SEQ[(sequences/index.json)]
    CLS --> TA[training/train_a.py<br/>YOLO-cls]
    YOLO --> TA2[training/train_a.py<br/>YOLO detector]
    TA --> EXP[export.py → ONNX]
    TA2 --> EXP
    EXP --> FC[data/sequences.py<br/>feature cache, synthetic layer shift]
    SEQ --> FC
    FC --> TB[training/train_b.py<br/>TCN / Transformer]
    TB --> EV[evaluation/*<br/>A vs B, delay, false alarms, FPS]
    TA --> EV
    TA2 --> EV
```

### Splits

| Dataset | Grouping unit | Rule |
|---|---|---|
| Trier FFF | contiguous frame block (pseudo-print; the archive has no print IDs) | official `test` archive = test; base blocks → train / val (20 %), stratified per class |
| YOLO-Strings | recording sequence number | val = sequences 17, 18, 20, 23, 25 (authors' own validation prints); test = 5, 9, 22, 28; FLIR + microscope views and Roboflow copies stay together |

`prepare.py` writes `splits.csv` and raises `RuntimeError("Leakage ...")` if any group appears in two
splits (tested in `tests/test_prepare.py`).

### Synthetic gradual layer shift

`printguard/data/augment.py` samples an onset (30–60 % into the sequence), a ramp length (8–25 frames)
and a maximum horizontal shift (3–8 % of the width). It estimates the row where new material is being
deposited and translates everything above it by a growing offset, producing per-frame labels that are
`0` before onset and `1` once the shift is visible. Clean sequences are reused as negatives, so the
comparison is not confounded by scene identity.

## Per-frame feature vector (Model B input)

| Block | Size | Source |
|---|---|---|
| Image embedding | 128 | frozen ImageNet MobileNetV3-Small, pooled 576-d → fixed random projection |
| Detector summary | 3 × detector classes | max confidence, box count (capped at 10, scaled), summed box-area fraction per class |
| Classifier probabilities | number of classifier classes | Model A classifier softmax |

Features are standardised with statistics fitted on the training split only (`FeatureNormalizer`).
Model B is **causal** (a prediction at time *t* only uses frames ≤ *t*), so it can run online.

## Online system

```mermaid
sequenceDiagram
    participant Cam as Camera / RTSP / file
    participant W as InferenceWorker (thread)
    participant P as OnlinePipeline
    participant DB as Timeline (SQLite)
    participant B as Broadcaster
    participant UI as Dashboard
    participant Op as Operator
    participant G as PauseGuard
    participant Pr as OctoPrint / Moonraker

    Cam->>W: frame (rate-limited to target_fps)
    W->>P: frame
    P-->>W: detections, class probs, temporal scores, trends
    W->>DB: events (score ≥ 0.5), alerts (k-of-n rule)
    W->>B: frame JPEG + event + alerts
    B-->>UI: WebSocket /ws (latest-only queue per client)
    Op->>UI: approve pause for alert #id
    UI->>G: POST /api/alerts/{id}/approve-pause (operator name)
    alt disabled, alert not open, or rate limited
        G-->>UI: 4xx refused (alert stays open)
    else dry-run
        G->>DB: audit row (dry_run=1) + alert → pause_requested
    else live
        G->>Pr: pause (timeout)
        G->>DB: audit row (ok / failed) + alert status
    end
```

### Threading and shutdown

- The inference worker runs in a daemon thread; the FastAPI event loop owns the WebSocket clients.
  `Broadcaster` hands messages across with `loop.call_soon_threadsafe` and a bounded, latest-only
  queue per client, so a slow browser never blocks inference.
- `Timeline` wraps one SQLite connection with a lock. On shutdown (e.g. SIGTERM in Docker), the
  lifespan handler stops the worker first, then closes the timeline. Writes that arrive after close, or
  fail because the database became read-only during teardown, are logged and dropped instead of
  crashing (`tests/test_db_shutdown.py`).

### Alert policy

`printguard/inference/alerts.py`: an alert opens when *k* of the last *n* scores for a label exceed the
threshold (default 3 of 5 above 0.5 – the same rule as the offline metrics), then that label is muted
for `cooldown_frames`. Temporal scores are used when Model B is loaded, otherwise Model A scores.

### Pause safety state machine

```mermaid
stateDiagram-v2
    [*] --> open: alert raised
    open --> dismissed: operator dismisses
    open --> pause_requested: operator approves & guard passes
    open --> pause_failed: operator approves & printer call fails
    open --> open: approve refused (disabled / rate limit)
    dismissed --> [*]
    pause_requested --> [*]
    pause_failed --> [*]
```

Guard checks, in order: pausing enabled in config → alert exists and is still `open` → minimum time
since the last real pause → dry-run (record only) or real request with timeout. Every executed attempt
(dry-run, success or failure) is written to `pause_requests`; refused requests leave the alert open.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | dashboard |
| GET | `/api/health` | liveness |
| GET | `/api/status` | worker, models, FPS, safety settings |
| POST | `/api/source` | start a `file` / `usb` / `rtsp` source |
| POST | `/api/source/stop` | stop the source |
| GET | `/api/events` | timeline events |
| GET | `/api/alerts` | alerts |
| POST | `/api/alerts/{id}/dismiss` | dismiss an alert |
| POST | `/api/alerts/{id}/approve-pause` | operator-approved pause (safety-gated) |
| GET | `/api/pause-requests` | audit log |
| GET | `/api/frame.jpg` | latest processed frame |
| WS | `/ws` | live frames, events, alerts |

Mutating endpoints require `Authorization: Bearer $PRINTGUARD_TOKEN` when that variable is set.
