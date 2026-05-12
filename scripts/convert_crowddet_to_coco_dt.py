#!/usr/bin/env python3
"""
Convert CrowdDet dump JSON(JSONL) to COCO detection list.

Input (CrowdDet): records with fields:
  - ID: image key (often file stem, e.g. "273271,1a0d6000b9e1f5b5")
  - dtboxes: list of dicts with
      - box: [x, y, w, h]
      - score: float
      - tag: int (optional)

Output (COCO-DT):
  [{"image_id": ..., "category_id": ..., "bbox": [x,y,w,h], "score": ...}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _norm_key(v: Any) -> str:
    return str(v).strip()


def _filename_keys(file_name: str) -> set[str]:
    keys: set[str] = set()
    p = Path(str(file_name))
    raw = str(file_name).strip()
    if raw:
        keys.add(raw)
    name = p.name
    if name:
        keys.add(name)
    stem = p.stem
    if stem:
        keys.add(stem)
    return keys


def _parse_input_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        return []

    if stripped[0] == "[":
        obj = json.loads(text)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        raise SystemExit("Expected JSON array for --crowddet-json")

    if stripped[0] == "{":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            if "Extra data" not in str(e):
                raise SystemExit(f"Invalid JSON in --crowddet-json: {e}") from e
        else:
            if isinstance(obj, dict) and "annotations" in obj and isinstance(obj["annotations"], list):
                return [x for x in obj["annotations"] if isinstance(x, dict)]
            if isinstance(obj, dict) and ("dtboxes" in obj or "ID" in obj):
                return [obj]
            raise SystemExit("Unsupported single JSON object for --crowddet-json")

    records: list[dict[str, Any]] = []
    for i, line in enumerate(text.splitlines()):
        ln = line.strip()
        if not ln:
            continue
        try:
            item = json.loads(ln)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSONL at line {i + 1}: {e}") from e
        if isinstance(item, dict):
            records.append(item)
    return records


def _build_image_map(coco_gt: dict[str, Any]) -> tuple[dict[str, int], list[str]]:
    images = coco_gt.get("images")
    if not isinstance(images, list):
        raise SystemExit("GT JSON must contain images[]")

    key_to_id: dict[str, int] = {}
    collisions: list[str] = []
    for im in images:
        if not isinstance(im, dict) or "id" not in im:
            continue
        img_id = int(im["id"])
        keys = {_norm_key(img_id)}
        file_name = im.get("file_name")
        if file_name is not None:
            keys |= _filename_keys(str(file_name))

        for k in keys:
            if k in key_to_id and key_to_id[k] != img_id:
                collisions.append(k)
                continue
            key_to_id[k] = img_id
    return key_to_id, collisions


def _resolve_image_id(rec_id: Any, key_to_id: dict[str, int]) -> int | None:
    key = _norm_key(rec_id)
    if key in key_to_id:
        return key_to_id[key]
    p = Path(key)
    if p.name in key_to_id:
        return key_to_id[p.name]
    if p.stem in key_to_id:
        return key_to_id[p.stem]
    return None


def _category_id_from_box(
    box: dict[str, Any],
    *,
    mode: str,
    fixed_category_id: int,
    skip_background_tag: bool,
) -> int | None:
    if mode == "fixed":
        return fixed_category_id
    tag = int(box.get("tag", fixed_category_id))
    if skip_background_tag and tag <= 0:
        return None
    return tag


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crowddet-json", type=Path, required=True, help="CrowdDet dump json/jsonl")
    ap.add_argument("--gt-json", type=Path, required=True, help="COCO GT instances json")
    ap.add_argument("--out-json", type=Path, required=True, help="Output COCO-DT json list")
    ap.add_argument("--score-thr", type=float, default=0.0, help="Filter detections by score")
    ap.add_argument(
        "--category-mode",
        choices=["tag", "fixed"],
        default="tag",
        help="Use CrowdDet 'tag' as category_id, or force fixed category_id",
    )
    ap.add_argument(
        "--category-id",
        type=int,
        default=1,
        help="Fixed category_id (used for --category-mode=fixed and as fallback)",
    )
    ap.add_argument(
        "--skip-background-tag",
        action="store_true",
        help="When --category-mode=tag, skip detections with tag <= 0",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a record ID cannot be mapped to GT image_id",
    )
    args = ap.parse_args()

    src_path = args.crowddet_json.expanduser().resolve()
    gt_path = args.gt_json.expanduser().resolve()
    out_path = args.out_json.expanduser().resolve()

    if not src_path.is_file():
        raise SystemExit(f"Input not found: {src_path}")
    if not gt_path.is_file():
        raise SystemExit(f"GT not found: {gt_path}")

    records = _parse_input_records(src_path)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    key_to_id, collisions = _build_image_map(gt)

    if collisions:
        uniq = sorted(set(collisions))
        print(
            f"WARNING: {len(uniq)} ambiguous image keys in GT (using first mapping), first: {uniq[:10]}",
            file=sys.stderr,
        )

    dt: list[dict[str, Any]] = []
    unmapped = 0
    seen_records = 0
    seen_boxes = 0
    skipped_by_score = 0
    skipped_by_cat = 0

    for rec in records:
        if "dtboxes" not in rec:
            continue
        seen_records += 1
        image_id = _resolve_image_id(rec.get("ID"), key_to_id)
        if image_id is None:
            unmapped += 1
            if args.strict:
                raise SystemExit(f"Cannot map record ID={rec.get('ID')!r} to any GT images[].id")
            continue

        boxes = rec.get("dtboxes") or []
        if not isinstance(boxes, list):
            continue
        for box in boxes:
            if not isinstance(box, dict):
                continue
            seen_boxes += 1
            score = float(box.get("score", 0.0))
            if score < args.score_thr:
                skipped_by_score += 1
                continue
            bbox = box.get("box")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue

            category_id = _category_id_from_box(
                box,
                mode=args.category_mode,
                fixed_category_id=int(args.category_id),
                skip_background_tag=bool(args.skip_background_tag),
            )
            if category_id is None:
                skipped_by_cat += 1
                continue

            dt.append(
                {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "score": score,
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dt, indent=2), encoding="utf-8")

    print(
        "[convert_crowddet_to_coco_dt] "
        f"records_with_dtboxes={seen_records}, boxes_seen={seen_boxes}, dt_written={len(dt)}, "
        f"unmapped_records={unmapped}, skipped_by_score={skipped_by_score}, skipped_by_category={skipped_by_cat}",
        file=sys.stderr,
    )
    print(f"Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
