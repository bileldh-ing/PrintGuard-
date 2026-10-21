# Contributing

Thanks for your interest in PrintGuard AI. Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/bileldh-ing/PrintGuard-.git printguard && cd printguard
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-train.txt
make test lint
```

The test suite runs on CPU, downloads nothing, and finishes in about a minute. CI only needs CPU
PyTorch plus `requirements.txt`, `pytest` and `httpx`.

## Workflow

1. Open an issue describing the bug or feature (for model changes, say how you will evaluate it).
2. Create a branch: `feat/<topic>`, `fix/<topic>`, `docs/<topic>`.
3. Keep changes focused; add or update tests in `tests/`.
4. Run `make format && make lint && make test` before pushing.
5. Use [Conventional Commits](https://www.conventionalcommits.org/) messages
   (`feat(server): ...`, `fix(data): ...`, `docs: ...`).

## Code style

- Python ≥ 3.10, type hints on public functions, docstrings on modules and public classes.
- `ruff check` and `ruff format` (config in `pyproject.toml`, line length 120).
- Configuration lives in `configs/*.yaml`; avoid hard-coded paths and magic numbers.

## Results and data

- **Never commit** datasets, model weights (`*.pt`, `*.onnx`), `runs/`, or anything over a few MB.
- Report metrics only from `make eval` outputs, with the profile (`smoke` / `full`), hardware and
  commit hash. Smoke-profile numbers must be labelled as such.
- Splits must stay grouped by print / sequence; `printguard.data.prepare` enforces this.

## Safety-related changes

Any change touching `server/printer.py` or the approve-pause endpoint must keep these invariants and
their tests: pause disabled and dry-run by default, explicit operator approval of one open alert,
rate limiting, audit logging, no automatic pause path.

## Security

Do not open public issues for vulnerabilities; contact the maintainer through GitHub instead. Never
commit API keys — printer keys are read from environment variables.
