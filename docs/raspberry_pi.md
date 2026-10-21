# Running PrintGuard AI on a Raspberry Pi

> **Status:** guide written from the torch-free Docker / ONNX path, **not yet measured on real Pi
> hardware**. The only measured number so far is ~13.8 FPS for the full per-frame pipeline on a 2-core
> aarch64 cloud CPU (`results/smoke/benchmark_cpu.json`). Pi numbers will be added once measured.

## Recommended hardware

- Raspberry Pi 5 (4 GB or 8 GB) or Pi 4 (4 GB+), 64-bit Raspberry Pi OS (Bookworm).
- Active cooling (sustained inference throttles a passively cooled Pi).
- USB UVC webcam or Pi Camera Module 3, pointed at the build plate.
- The Pi can be the same one running OctoPrint / Klipper + Moonraker, or a separate one.

A print monitor needs roughly one frame per printed layer (a few seconds per frame), so even a few FPS
is plenty. Use `source.target_fps` to cap the inference rate and leave CPU for the printer host.

## Option A — Docker (recommended)

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker

git clone https://github.com/bileldh-ing/PrintGuard-.git printguard && cd printguard
mkdir -p weights data
# copy the ONNX exports produced by `make export` (or the Colab notebooks) into ./weights:
#   model_a_detect.onnx  model_a_classify.onnx  embedder.onnx
docker compose up -d --build
```

Open `http://<pi-hostname>:8000`. For a USB camera, uncomment the `devices:` block in
`docker-compose.yml` (`/dev/video0`) and start a `usb` source with URI `0` from the dashboard.

The Docker config (`configs/server.docker.yaml`) runs **Model A only** on ONNX Runtime, because Model B
checkpoints are PyTorch today. To run Model B on the Pi, use option B with PyTorch installed.

## Option B — native Python

```bash
sudo apt install -y python3-venv libgl1 libglib2.0-0
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # torch-free runtime
# optional, for Model B on the Pi (aarch64 wheels exist on PyPI):
# pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m server --config configs/server.docker.yaml
```

Run it as a service:

```ini
# /etc/systemd/system/printguard.service
[Unit]
Description=PrintGuard AI
After=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/printguard
Environment=PRINTGUARD_TOKEN=change-me
Environment=OCTOPRINT_API_KEY=your-key
ExecStart=/home/pi/printguard/.venv/bin/python -m server --config configs/server.docker.yaml
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now printguard
```

## Connecting the printer (pause stays manual by default)

```yaml
printer:
  pause_enabled: true      # still requires operator approval of each alert
  dry_run: true            # keep true until you have watched it on a few prints
  backend: octoprint       # or moonraker
  url: http://localhost    # OctoPrint on the same Pi; Moonraker usually http://localhost:7125
  api_key_env: OCTOPRINT_API_KEY
```

Set `PRINTGUARD_TOKEN` whenever the dashboard is reachable from your network.

## Performance tips

- Lower `server.max_width` (e.g. 640) and the detector `imgsz` (e.g. 320 or 416) after exporting at that size.
- Lower `source.target_fps` to 1–2.
- Prefer a fixed camera mount and stable lighting; both matter more than model size.

## Measuring

```bash
python -m printguard.evaluation.benchmark --detector weights/model_a_detect.onnx \
  --classifier weights/model_a_classify.onnx --embedder weights/embedder.onnx --out results/pi_benchmark.json
```

Please open a PR with the JSON if you measure on a Pi.
