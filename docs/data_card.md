# Data card

PrintGuard AI does not redistribute any data. `make data` downloads from Zenodo, verifies checksums
and builds the processed datasets locally under `data/` (git-ignored).

## Sources

### Trier FFF — Bast et al. 2026

- **Citation:** Bast, S., Scherer, K., Wahl, M., Naumann, S., Dartmann, G., Didas, S. (2026). *Quality
  Control in Additive Manufacturing: Image Dataset for Defect Detection in Fused Filament Fabrication*
  (v2.0.0). Zenodo. https://doi.org/10.5281/zenodo.18628777
- **License:** CC-BY-4.0
- **Content:** images of FFF prints from three camera angles (`camera1–3`), image-level classes
  extrusion error, layer shift, no error; an additional archive with stringing / no-stringing images.
  `base_data.zip`, `test_data.zip`, `additional_defect_data.zip`.
- **Used for:** Model A frame classifier (`extrusion_error`, `layer_shift`, `no_error`); clean frame
  sequences as the base for synthetic gradual layer shift (Model B).
- **Class mapping:** `extrusion_error_print → extrusion_error`, `layershift_error_print → layer_shift`,
  `no_error_print → no_error`, `stringing_error → stringing`, `no_stringing_error → no_error`.

### YOLO-Strings — Charia et al. 2025

- **Citation:** Charia, O., Rajani, H., Domingo-Espin, M., Ferrer Real, I., Gracias, N. (2025).
  *YOLO-Strings: A YOLO-compatible Dataset for Detecting Stringing during 3D Printing.* Zenodo.
  https://doi.org/10.5281/zenodo.14711320
- **License:** CC-BY-4.0
- **Content:** YOLO-format bounding boxes for stringing (Roboflow export with augmented copies), plus
  per-print image sequences from a microscope camera and a FLIR camera, grouped into
  `with_stringing` / `without_stringing` sequences; roughly one frame per printed layer.
- **Used for:** Model A stringing detector; real stringing sequences for Model B.

## Download and integrity

`printguard/data/download.py` uses HTTP range requests (via `remotezip`) to fetch only the archive
members a profile needs, verifying each member's CRC-32. `--whole-archives` downloads complete ZIPs and
checks Zenodo's published MD5. Record manifests are cached in `data/manifests/`.

## Profiles

| Profile | Purpose | Size | Notes |
|---|---|---|---|
| `smoke` | CPU sandbox / CI-sized runs | a few hundred MB | 6 base + 2 test blocks of 12 frames per class × camera from Trier; 260 YOLO-Strings source images; a fixed list of microscope sequences; images downscaled to 480 px |
| `full` | GPU training | tens of GB | every frame, including the Trier stringing archive; all YOLO-Strings sequences |

## Splits

- **Trier:** the official test archive is the test set. Base frames are grouped into contiguous blocks
  (a stand-in for a print, since the archive has no print IDs); whole blocks go to train or val
  (`val_fraction: 0.2`), stratified per class.
- **YOLO-Strings:** split by recording sequence number. Val = 17, 18, 20, 23, 25; test = 5, 9, 22, 28.
  A print's microscope and FLIR views and all Roboflow augmented copies of a source image stay in one
  split.
- `prepare.py` writes `data/processed/splits.csv` and refuses to finish if a group leaks across splits.

## Preprocessing

- Labels converted to YOLO format (`yolo_strings/dataset.yaml`); classification data arranged as
  `trier_cls/{split}/{class}/`.
- Sequence index (`sequences/index.json`) stores ordered frame paths (numeric frame order) and per-frame
  labels, with `-1` meaning "unknown" (masked out of the loss).
- Synthetic gradual layer-shift variants are generated on the fly while building the feature cache,
  with per-sequence deterministic seeds.

## Known limitations and biases

- **Scene identity correlates with label** in YOLO-Strings sequences (stringing and clean sequences are
  different prints), which temporal models can exploit.
- **No real layer-shift sequences:** Model B's layer-shift labels are synthetic.
- **Pseudo-print blocks** in Trier are an approximation; neighbouring blocks may still come from the same
  physical print.
- **Limited diversity:** few printers, filament colours and lighting setups; no spaghetti or warping.
- **No timing metadata:** layer durations are unknown, so per-hour rates use an assumed seconds-per-frame.

## Personal data

None known. The images show printed parts and printer hardware only.
