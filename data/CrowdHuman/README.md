# CrowdHuman unified eval snapshot

`crowddet_unified_metrics_epoch30.json` is a frozen copy of `scripts/eval_coco_predictions.py` output for **RCNN EMD Refine**, epoch **30**, CrowdHuman val.

**Run context (2026-05):** GPU instance; `val.json` from `freeyolo_prepare_crowdhuman.py` bridge; `dump-30.json` from `tools/test.py -md rcnn_emd_refine -r 30`; converter `--score-thr 0.0 --category-mode fixed --category-id 1 --strict`. Native CrowdHuman AP from the same run was about **0.902** (different protocol than COCO `AP50` here).

Re-run on your machine to regenerate; this file is for reporting and diffing against the benchmark study repo.
