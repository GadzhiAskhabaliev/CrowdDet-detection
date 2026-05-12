# Unified COCO-style evaluation (CrowdHuman val)

This fork adds tooling so CrowdDet dumps can be scored with the same **pycocotools `COCOeval`** pipeline used in the pedestrian benchmark repo ([real-time-people-detection-and-tracking-on-edge](https://github.com/GadzhiAskhabaliev/real-time-people-detection-and-tracking-on-edge)): `scripts/eval_coco_predictions.py`, metric keys, and `--strict` image-id checks ([unified COCOeval notes](https://github.com/GadzhiAskhabaliev/real-time-people-detection-and-tracking-on-edge/blob/main/docs/benchmark_unified_cocoeval.md), [remote MMDet / CrowdDet bridge](https://github.com/GadzhiAskhabaliev/real-time-people-detection-and-tracking-on-edge/blob/main/docs/group_b_remote_mmdet_bridge.md)).

Native CrowdHuman numbers from `tools/test.py` (AP / MR / JI) stay unchanged; unified AP is **not** identical to that protocol.

## Prerequisites

- Python env with **PyTorch**, **OpenCV**, **tqdm**, **numpy**, **pycocotools** (see root `requirements.txt` plus `pip install pycocotools`).
- CrowdHuman **val** images + `annotation_val.odgt` on disk; paths in `model/<config>/config.py` (default layout under `/data/CrowdHuman/`).
- CrowdDet checkpoint as `model/<model_dir>/outputs/model_dump/dump-<epoch>.pth` (see upstream README for weights).
- A COCO-instances **`val.json`** whose `images[].id` and `file_name` align with the same val split used for other models in your benchmark (build once; do not regenerate with a different id assignment).

### `val.json` from odgt + jpg (bridge-compatible)

The benchmark repo builds a bridge tree with `scripts/group_b/freeyolo_prepare_crowdhuman.py` (expects `Images/*.jpg` and `annotation_val.odgt`). This repo loads **`{ID}.png`**; if your files are **`.jpg`**, create symlinks:

```bash
cd /path/to/crowdhuman/images
for f in *.jpg; do ln -sf "$f" "${f%.jpg}.png"; done
```

Then run `freeyolo_prepare_crowdhuman.py` (from the benchmark repo clone) with `--crowdhuman-root` pointing at that tree. It writes `.../CrowdHuman/annotations/val.json`.

## 1) Native evaluation (upstream)

From repo root:

```bash
cd tools
python3 test.py -md rcnn_emd_refine -r 30 -d 0
```

Outputs include `model/rcnn_emd_refine/outputs/eval_dump/dump-<epoch>.json` (JSONL: **one JSON object per line**) and `eval-<epoch>.json`.

## 2) Convert CrowdDet dump → COCO detection list

```bash
python3 scripts/convert_crowddet_to_coco_dt.py \
  --crowddet-json model/rcnn_emd_refine/outputs/eval_dump/dump-30.json \
  --gt-json /path/to/val.json \
  --out-json /path/to/crowddet_dt.json \
  --category-mode fixed \
  --category-id 1 \
  --score-thr 0.0 \
  --strict
```

Each output row: `image_id`, `category_id`, `bbox` `[x,y,w,h]`, `score`. The converter maps CrowdDet record `ID` to COCO `image_id` using `val.json` (`id`, `file_name`, basename, stem).

## 3) Unified metrics

```bash
python3 scripts/eval_coco_predictions.py \
  --gt-json /path/to/val.json \
  --dt-json /path/to/crowddet_dt.json \
  --strict \
  --out-metrics-json /path/to/metrics.json \
  --out-patch-json /path/to/patch.json
```

Optional: merge `patch.json` into a benchmark run via `bench_runner.py --merge-json ...` in the study repo.

## Troubleshooting

| Issue | What to do |
|--------|----------------|
| `JSONDecodeError: Extra data` on convert | Use the **updated** `convert_crowddet_to_coco_dt.py` in this fork (JSONL vs single JSON). |
| `image_id` not in GT with `--strict` | Same `val.json` as dump mapping; check `ID` vs `file_name` / order of odgt when building `val.json`. |
| Missing `.png` | Symlink `.png` → `.jpg` per image id, or change `lib/data/CrowdHuman.py` (not recommended for staying close to upstream). |
| Google Drive download fails on cloud GPU | Prefer Hugging Face mirrors, e.g. `scripts/vast/download_crowdhuman_val.sh` in the benchmark repo ([HF `sshao0516/CrowdHuman`](https://huggingface.co/datasets/sshao0516/CrowdHuman)). |

## What to log for reproducibility

- Git commit, model dir, `-r` epoch, `config.py` thresholds (`pred_cls_threshold`, NMS).
- Absolute path to **`val.json`** used for convert + eval.
- `--score-thr` passed to the converter (if non-zero).
