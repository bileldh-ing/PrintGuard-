# PrintGuard AI - common tasks. `make help` lists them.
PY        ?= python
PROFILE   ?= smoke          # smoke (CPU, minutes) | full (GPU recommended)
DEVICE    ?= cpu            # cpu | 0 (first CUDA GPU)
SMOKE_A_DET := $(if $(filter smoke,$(strip $(PROFILE))),--override configs/smoke/model_a_detect.yaml,)
SMOKE_A_CLS := $(if $(filter smoke,$(strip $(PROFILE))),--override configs/smoke/model_a_classify.yaml,)
SMOKE_B     := $(if $(filter smoke,$(strip $(PROFILE))),--override configs/smoke/model_b.yaml,)
WDIR        := $(if $(filter smoke,$(strip $(PROFILE))),weights/smoke,weights)
RDIR        := $(if $(filter smoke,$(strip $(PROFILE))),results/smoke,results)
DET_IMGSZ   := $(if $(filter smoke,$(strip $(PROFILE))),256,640)
CLS_IMGSZ   := $(if $(filter smoke,$(strip $(PROFILE))),128,224)

.PHONY: help install install-train data train-a train-b export eval benchmark serve demo test lint format docker clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:         ## runtime deps (server + ONNX inference)
	$(PY) -m pip install -r requirements.txt

install-train:   ## training deps (CPU torch wheel)
	$(PY) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PY) -m pip install -r requirements-train.txt

data:            ## download (CRC/MD5-checked) + prepare leakage-safe splits
	$(PY) -m printguard.data.download --profile $(strip $(PROFILE))
	$(PY) -m printguard.data.prepare --profile $(strip $(PROFILE))

train-a:         ## Model A: YOLO detector (YOLO-Strings) + frame classifier (Trier)
	$(PY) -m printguard.training.train_a --config configs/model_a_detect.yaml $(SMOKE_A_DET) --device $(strip $(DEVICE))
	$(PY) -m printguard.training.train_a --config configs/model_a_classify.yaml $(SMOKE_A_CLS) --device $(strip $(DEVICE))

export:          ## ONNX export of Model A + feature embedder
	$(PY) -m printguard.export yolo --weights $(WDIR)/model_a_detect.pt --imgsz $(DET_IMGSZ)
	$(PY) -m printguard.export yolo --weights $(WDIR)/model_a_classify.pt --imgsz $(CLS_IMGSZ)
	$(PY) -m printguard.export embedder --out $(WDIR)/embedder.onnx

train-b: export  ## Model B: features (resumable) + TCN + Transformer
	$(PY) -m printguard.training.train_b features --config configs/model_b_tcn.yaml $(SMOKE_B)
	$(PY) -m printguard.training.train_b train --config configs/model_b_tcn.yaml $(SMOKE_B)
	$(PY) -m printguard.training.train_b train --config configs/model_b_transformer.yaml $(SMOKE_B)

eval:            ## test-split metrics for A and B + FPS benchmark
	$(PY) -m printguard.evaluation.eval_a classifier --weights $(WDIR)/model_a_classify.onnx --out $(RDIR)/model_a
	$(PY) -m printguard.evaluation.eval_a detector --weights $(WDIR)/model_a_detect.pt --imgsz $(DET_IMGSZ) --out $(RDIR)/model_a
	$(PY) -m printguard.training.train_b evaluate --config configs/model_b_tcn.yaml $(SMOKE_B)
	$(PY) -m printguard.training.train_b evaluate --config configs/model_b_transformer.yaml $(SMOKE_B)
	$(MAKE) benchmark PROFILE=$(strip $(PROFILE))

benchmark:       ## CPU latency / FPS of every pipeline stage
	$(PY) -m printguard.evaluation.benchmark --detector $(WDIR)/model_a_detect.onnx --det-imgsz $(DET_IMGSZ) \
	  --classifier $(WDIR)/model_a_classify.onnx --embedder $(WDIR)/embedder.onnx \
	  --temporal $(WDIR)/model_b_tcn.pt --out $(RDIR)/benchmark_cpu.json

demo:            ## build a replay video from a held-out test sequence (data/demo/*.mp4)
	$(PY) scripts/make_demo_video.py

serve:           ## API + dashboard on http://localhost:8000
	$(PY) -m server $(if $(filter smoke,$(strip $(PROFILE))),--config configs/server.smoke.yaml,)

test:            ## unit + API tests
	$(PY) -m pytest

lint:            ## ruff lint + format check
	ruff check . && ruff format --check .

format:          ## auto-format
	ruff check --fix . && ruff format .

docker:          ## build and run with docker compose
	docker compose up --build

clean:           ## remove caches (keeps data and weights)
	rm -rf .pytest_cache .ruff_cache runs **/__pycache__
