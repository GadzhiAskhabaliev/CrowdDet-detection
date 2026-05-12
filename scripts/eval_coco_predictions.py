#!/usr/bin/env python3
"""
Evaluate bbox detection: COCO-instances **GT** JSON + list of **DT** dicts.

GT is any single split (e.g. CrowdHuman val) with `images[].id`. DT rows must use
those `image_id` values. Each DT dict: `image_id`, `category_id`, `bbox` [x,y,w,h]
pixels xywh, `score`.

Outputs: AP25 / AP50 / AP75 / AP50-95 (pycocotools COCOeval bbox; single-class = no extra
"mean" beyond COCO's per-category average); **COCO-official**
AR (maxDets=100) at IoU 0.25 / 0.50 / 0.75 plus AR IoU=0.50:0.95 as `recall`; **COCO**
precision at recall grid 0.5 on the AP curve at those IoUs + FDR=1-precision; and
optional greedy micro P/R/FDR at `--precision-score-thr` (legacy `precision` = greedy
@ `--precision-iou-thr`). No FPS.

  python3 scripts/eval_coco_predictions.py --gt-json .../val.json --dt-json .../dt.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np


def _xywh_to_xyxy(b: list[float] | tuple[float, ...]) -> tuple[float, float, float, float]:
    x, y, w, h = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    return x, y, x + w, y + h


def _iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _greedy_by_image_dets(
    raw_dt: list[dict[str, Any]], *, score_thr: float
) -> dict[int, list[dict[str, Any]]]:
    by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for d in raw_dt:
        if float(d["score"]) < score_thr:
            continue
        by_img[int(d["image_id"])].append(d)
    for img_id in by_img:
        by_img[img_id].sort(key=lambda x: float(x["score"]), reverse=True)
    return by_img


def _greedy_micro_prf(
    coco_gt: Any,
    by_img: dict[int, list[dict[str, Any]]],
    *,
    iou_thr: float,
) -> tuple[float, float, float]:
    """
    Micro precision = TP/(TP+FP), recall = TP/N_gt, FDR = 1 - precision over all GT
    images: each scored detection is TP or FP; N_gt counts non-crowd boxes on every
    GT image. Greedy match per image (same category_id), IoU >= iou_thr.
    """
    tp = 0
    fp = 0
    total_gt = 0
    for img_id in coco_gt.getImgIds():
        ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        anns = coco_gt.loadAnns(ann_ids)
        gts: list[dict[str, Any]] = []
        for a in anns:
            if a.get("iscrowd", 0) == 1:
                continue
            gts.append(
                {
                    "bbox": a["bbox"],
                    "cat": int(a["category_id"]),
                    "xyxy": _xywh_to_xyxy(a["bbox"]),
                }
            )
        total_gt += len(gts)
        dets = by_img.get(img_id, [])
        matched = [False] * len(gts)

        for det in dets:
            dxy = _xywh_to_xyxy(det["bbox"])
            dcat = int(det["category_id"])
            best_j = -1
            best_iou = 0.0
            for j, g in enumerate(gts):
                if matched[j] or g["cat"] != dcat:
                    continue
                iou = _iou_xyxy(dxy, g["xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                matched[best_j] = True
                tp += 1
            else:
                fp += 1

    denom_pr = tp + fp
    prec = float(tp / denom_pr) if denom_pr > 0 else 0.0
    rec = float(tp / total_gt) if total_gt > 0 else 0.0
    fdr = 1.0 - prec
    return prec, rec, fdr


def _aind_mind_all_100(coco_eval: Any) -> tuple[int, int]:
    p = coco_eval.params
    aind = next(i for i, lbl in enumerate(p.areaRngLbl) if lbl == "all")
    mind = next(i for i, md in enumerate(p.maxDets) if md == 100)
    return aind, mind


def _mean_ap_from_eval_precision(coco_eval: Any, *, iou_ix: int) -> float:
    """Mean AP slice matching COCO summarize (area=all, maxDets=100)."""
    aind, mind = _aind_mind_all_100(coco_eval)
    s = coco_eval.eval["precision"][iou_ix, :, :, aind, mind]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else 0.0


def _iou_thr_to_index(coco_eval: Any, iou_thr: float) -> int:
    p = coco_eval.params
    t = np.flatnonzero(np.isclose(p.iouThrs, float(iou_thr), rtol=0.0, atol=1e-9))
    if t.size != 1:
        raise ValueError(f"IoU {iou_thr} not in coco_eval.params.iouThrs")
    return int(t[0])


def _mean_coco_ar_at_iou(coco_eval: Any, *, iou_thr: float) -> float:
    """COCO AR (max recall over score ranking), area=all, maxDets=100, single IoU slice."""
    t = _iou_thr_to_index(coco_eval, iou_thr)
    aind, mind = _aind_mind_all_100(coco_eval)
    s = coco_eval.eval["recall"][t, :, aind, mind]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else 0.0


def _mean_coco_precision_at_recall(
    coco_eval: Any, *, iou_thr: float, recall_grid: float
) -> float:
    """
    Mean precision on the official COCO PR grid at one recall threshold (default 0.5),
    area=all, maxDets=100. Same tensor COCO uses for AP; not the same as greedy P@fixed
    score.
    """
    p = coco_eval.params
    r_ix = int(np.round(float(recall_grid) / 0.01))
    if r_ix < 0 or r_ix >= len(p.recThrs):
        raise ValueError(f"recall_grid {recall_grid} out of recThrs range")
    t = _iou_thr_to_index(coco_eval, iou_thr)
    aind, mind = _aind_mind_all_100(coco_eval)
    s = coco_eval.eval["precision"][t, r_ix, :, aind, mind]
    s = s[s > -1]
    return float(np.mean(s)) if s.size else 0.0


def _validate_entries(raw_dt: list[Any]) -> None:
    need = ("image_id", "category_id", "bbox", "score")
    for i, d in enumerate(raw_dt):
        if not isinstance(d, dict):
            raise SystemExit(f"dt[{i}] must be a dict object")
        for k in need:
            if k not in d:
                raise SystemExit(f"dt[{i}]: missing key {k!r}")
        b = d["bbox"]
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            raise SystemExit(f"dt[{i}]: bbox must be [x,y,w,h] with four numbers")


def _check_image_ids(coco_gt: Any, raw_dt: list[dict[str, Any]], *, strict: bool) -> None:
    gt_ids = set(coco_gt.getImgIds())
    dt_ids = {int(d["image_id"]) for d in raw_dt}
    unknown = sorted(dt_ids - gt_ids)
    if not unknown:
        return
    msg = (
        f"Predictions contain image_id not in GT ({len(unknown)} ids), "
        f"first: {unknown[:10]}{'...' if len(unknown) > 10 else ''}. "
        "GT and DT refer to different splits or annotation files."
    )
    if strict:
        raise SystemExit(msg)
    print(f"WARNING: {msg}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-json", type=Path, required=True, help="COCO GT (instances)")
    p.add_argument(
        "--dt-json",
        type=Path,
        required=True,
        help="COCO-style detection list or mmcv/json dump of a list",
    )
    p.add_argument(
        "--out-metrics-json",
        type=Path,
        default=None,
        help="Write metrics object (AP*, COCO AR/P/FDR, optional greedy P/R/FDR)",
    )
    p.add_argument(
        "--precision-score-thr",
        type=float,
        default=0.5,
        help="Score threshold for greedy precision (default 0.5)",
    )
    p.add_argument(
        "--precision-iou-thr",
        type=float,
        default=0.5,
        help="Legacy: single IoU for greedy precision only (default 0.5); multi-IoU uses --greedy-iou-thrs",
    )
    p.add_argument(
        "--greedy-iou-thrs",
        type=str,
        default="0.25,0.5,0.75",
        help="Comma-separated IoU thresholds for greedy precision/recall/FDR (default 0.25,0.5,0.75)",
    )
    p.add_argument(
        "--coco-pr-recall",
        type=float,
        default=0.5,
        help="Recall grid point (0..1) for official COCO precision / FDR from eval['precision'] (default 0.5)",
    )
    p.add_argument(
        "--out-patch-json",
        type=Path,
        default=None,
        help="Patch JSON for bench_runner --merge-json: {metrics, notes}",
    )
    p.add_argument(
        "--quiet-summarize",
        action="store_true",
        help="Do not print default pycocotools summarize() block",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with error if DT references unknown image_id vs GT",
    )
    args = p.parse_args()

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise SystemExit("Requires pycocotools: pip install pycocotools\n" + str(e)) from e

    gt_path = args.gt_json.expanduser().resolve()
    dt_path = args.dt_json.expanduser().resolve()
    if not gt_path.is_file():
        raise SystemExit(f"GT not found: {gt_path}")
    if not dt_path.is_file():
        raise SystemExit(f"DT not found: {dt_path}")

    raw_dt = json.loads(dt_path.read_text(encoding="utf-8"))
    if isinstance(raw_dt, dict) and "annotations" in raw_dt:
        raw_dt = raw_dt["annotations"]
    if not isinstance(raw_dt, list):
        raise SystemExit("dt-json must be a list of detections or dict with 'annotations'")

    _validate_entries(raw_dt)

    coco_gt = COCO(str(gt_path))
    _check_image_ids(coco_gt, raw_dt, strict=args.strict)

    n_gt_img = len(coco_gt.getImgIds())
    dt_img_hit = len({int(d["image_id"]) for d in raw_dt})
    print(
        f"[eval_coco_predictions] GT images: {n_gt_img}, "
        f"images with >=1 detection in DT: {dt_img_hit}, detections: {len(raw_dt)}",
        file=sys.stderr,
    )

    coco_dt = coco_gt.loadRes(raw_dt)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    if args.quiet_summarize:
        with redirect_stdout(io.StringIO()):
            coco_eval.summarize()
    else:
        coco_eval.summarize()

    stats = coco_eval.stats
    ap5095 = float(stats[0])
    ap50 = float(stats[1])
    ap75 = float(stats[2])
    ar100 = float(stats[8])

    coco_eval25 = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval25.params.iouThrs = np.array([0.25], dtype=np.float64)
    coco_eval25.evaluate()
    coco_eval25.accumulate()
    ap25 = _mean_ap_from_eval_precision(coco_eval25, iou_ix=0)

    try:
        greedy_ious = [float(x.strip()) for x in args.greedy_iou_thrs.split(",") if x.strip()]
    except ValueError as e:
        raise SystemExit(f"Invalid --greedy-iou-thrs: {args.greedy_iou_thrs!r}") from e
    if not greedy_ious:
        raise SystemExit("--greedy-iou-thrs must list at least one IoU")

    by_img = _greedy_by_image_dets(raw_dt, score_thr=args.precision_score_thr)
    pr_r = float(args.coco_pr_recall)
    if not (0.0 <= pr_r <= 1.0):
        raise SystemExit("--coco-pr-recall must be in [0, 1]")

    metrics: dict[str, Any] = {
        "AP25": round(ap25, 6),
        "AP50": round(ap50, 6),
        "AP75": round(ap75, 6),
        "AP50-95": round(ap5095, 6),
        "recall": round(ar100, 6),
    }

    r_lab = f"{int(round(pr_r * 100)):d}"
    coco_eval_slices: list[tuple[float, Any]] = [
        (0.25, coco_eval25),
        (0.5, coco_eval),
        (0.75, coco_eval),
    ]
    for iou_thr, ev in coco_eval_slices:
        tag = f"{int(round(float(iou_thr) * 100))}"
        metrics[f"coco_ar_iou{tag}"] = round(_mean_coco_ar_at_iou(ev, iou_thr=iou_thr), 6)
        prec_c = _mean_coco_precision_at_recall(ev, iou_thr=iou_thr, recall_grid=pr_r)
        metrics[f"coco_precision_r{r_lab}_iou{tag}"] = round(prec_c, 6)
        metrics[f"coco_fdr_r{r_lab}_iou{tag}"] = round(1.0 - prec_c, 6)

    for iou_thr in greedy_ious:
        prec_g, rec_g, fdr_g = _greedy_micro_prf(coco_gt, by_img, iou_thr=iou_thr)
        tag = f"{int(round(float(iou_thr) * 100))}"
        metrics[f"precision_iou{tag}"] = round(prec_g, 6)
        metrics[f"recall_iou{tag}"] = round(rec_g, 6)
        metrics[f"fdr_iou{tag}"] = round(fdr_g, 6)

    prec_legacy, _, fdr_legacy = _greedy_micro_prf(coco_gt, by_img, iou_thr=float(args.precision_iou_thr))
    metrics["precision"] = round(prec_legacy, 6)
    metrics["fdr"] = round(fdr_legacy, 6)

    print(json.dumps(metrics, indent=2))
    print(
        "\n(`recall` = COCO AR maxDets=100 IoU=0.50:0.95; `coco_ar_iou*` = official COCO AR "
        "at that single IoU, maxDets=100; `coco_precision_r*_iou*` / `coco_fdr_*` = mean "
        f"over categories from COCOeval PR tensor at recall={pr_r}, maxDets=100; "
        "AP25 from extra COCOeval IoU=[0.25]; AP50/AP75/AP50-95 from default summarize; "
        "precision_iou* / recall_iou* / fdr_iou* = greedy micro at "
        f"score>={args.precision_score_thr}, IoU in {greedy_ious}; "
        f"`precision`/`fdr` = greedy @ IoU={args.precision_iou_thr}. "
        "See docs/benchmark_metrics_schema.md)",
        file=sys.stderr,
    )

    if args.out_metrics_json:
        args.out_metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_metrics_json}", file=sys.stderr)

    if args.out_patch_json:
        patch = {
            "metrics": metrics,
            "notes": [
                "Quality: scripts/eval_coco_predictions.py — COCOeval bbox on "
                "--gt-json/--dt-json; recall=COCO AR maxDets=100 IoU=0.50:0.95; "
                "coco_ar_iou25/50/75 + coco_precision_r*_iou* + coco_fdr_* from official "
                f"pycocotools tensors (PR recall grid={pr_r}); AP25 extra eval IoU=[0.25]; "
                f"greedy P/R/FDR score>={args.precision_score_thr}, IoUs {greedy_ious}; "
                f"legacy precision/fdr greedy @ IoU={args.precision_iou_thr}."
            ],
        }
        args.out_patch_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_patch_json.write_text(json.dumps(patch, indent=2), encoding="utf-8")
        print(f"Wrote {args.out_patch_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
