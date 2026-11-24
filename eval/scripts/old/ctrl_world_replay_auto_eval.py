#!/usr/bin/env python3
"""
Automatic Ctrl-World replay evaluator.

No manually prepared CSV is required.

Expected inputs:
  --rollout-dir synthetic_traj/Rollouts_replay_droid
      evaluation_data/trajectory_<DROID_ID>/interaction_*.npz
  --dataset-root dataset_example/droid
      annotation/val/<DROID_ID>.json

The rollout script saves aligned interaction NPZ files containing:
  ground_truth, prediction, trajectory_id, interaction_index,
  start_frame_index, instruction, fps, frame_indices

This evaluator uses the generated `prediction` arrays, joins the DROID
annotation by trajectory_id, and compares the local VLM judgment with
annotation['success'].

IMPORTANT: DROID labels are valid for replay of the corresponding original
DROID trajectory. Do not use them as ground truth for a different imagined
policy-in-the-loop trajectory.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


PROMPT = """You are a careful evaluator of robot manipulation task completion.
You will receive a task instruction and temporally sampled images from a
GENERATED robot rollout. Each image may show multiple camera views.

Decide whether the generated rollout visibly completed the instruction.
Judge the requested outcome, not pixel similarity to a reference.
For placement/stacking, verify the requested final object relation.
For opening/closing, verify the final configuration.
Consider the temporal sequence and final state. If the evidence is ambiguous,
occluded, or insufficient, return uncertain. Gripper proximity alone is not
proof of success.

Return ONLY JSON:
{"verdict":"success|failure|uncertain","confidence":0.0,
 "evidence":"brief visual evidence","failure_reason":"brief reason or empty"}
"""


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout-dir", required=True,
                   help="Ctrl-World task output folder, e.g. synthetic_traj/Rollouts_replay_droid")
    p.add_argument("--dataset-root", required=True,
                   help="Matching DROID dataset folder, e.g. dataset_example/droid")
    p.add_argument("--annotation-split", default="val")
    p.add_argument("--db", default="task_completion_eval.sqlite")
    p.add_argument("--out-dir", default="task_completion_results")
    p.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--max-side", type=int, default=640)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--min-confidence", type=float, default=0.55)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def read_annotation(dataset_root, split, trajectory_id):
    path = Path(dataset_root) / "annotation" / split / f"{trajectory_id}.json"
    if not path.is_file():
        # Preserve zero padding if IDs are zero-padded in annotations.
        matches = list((Path(dataset_root) / "annotation" / split).glob(f"{int(trajectory_id):06d}.json")) if trajectory_id.isdigit() else []
        if matches:
            path = matches[0]
        else:
            return None, str(path)
    with path.open(encoding="utf-8") as f:
        anno = json.load(f)
    label = anno.get("success")
    if label not in (0, 1, False, True):
        return None, str(path)
    return int(label), str(path)


def discover_trajectories(rollout_dir, dataset_root, split):
    root = Path(rollout_dir)
    eval_root = root / "evaluation_data"
    if not eval_root.is_dir():
        raise FileNotFoundError(
            f"Could not find {eval_root}. Expected rollout_replay_traj.py output structure."
        )
    trajectories = []
    for traj_dir in sorted(eval_root.glob("trajectory_*")):
        if not traj_dir.is_dir():
            continue
        trajectory_id = traj_dir.name.removeprefix("trajectory_")
        npz_files = sorted(traj_dir.glob("interaction_*.npz"))
        if not npz_files:
            continue
        label, annotation_path = read_annotation(dataset_root, split, trajectory_id)
        # Load the instruction from the first available interaction NPZ.
        instruction = ""
        for fpath in npz_files:
            try:
                with np.load(fpath, allow_pickle=False) as z:
                    if "instruction" in z:
                        instruction = str(np.asarray(z["instruction"]).item())
                        if instruction:
                            break
            except Exception:
                continue
        if not instruction:
            # DROID annotation may contain the task text; prefer it if available.
            ap = Path(annotation_path)
            if ap.is_file():
                try:
                    instruction = str(json.loads(ap.read_text()).get("texts", [""])[0])
                except Exception:
                    instruction = ""
        if not instruction.strip():
            instruction = ("No language instruction is available for this DROID episode. "
                           "Infer the intended manipulation goal from the visual sequence, "
                           "then judge whether the episode appears successfully completed. "
                           "If the goal cannot be inferred, return uncertain.")
            print(f"[WARN] trajectory {trajectory_id} has no text instruction; using visual-only fallback.")
        trajectories.append({
            "episode_id": trajectory_id,
            "task_name": root.name,
            "instruction": instruction,
            "npz_files": [str(x) for x in npz_files],
            "annotation_path": annotation_path,
            "gt_success": label,
        })
    return trajectories


def npz_frames_for_trajectory(row, max_side):
    frames = []
    frame_ids = []
    for npz_path in row["npz_files"]:
        with np.load(npz_path, allow_pickle=False) as z:
            if "prediction" not in z:
                raise KeyError(f"'prediction' missing in {npz_path}")
            pred = np.asarray(z["prediction"])
            # rollout_replay_traj.py saves prediction as (camera, time, H, W, C).
            # Also accept (time,H,W,C) and (time,camera,H,W,C).
            if pred.ndim == 5 and pred.shape[-1] in (1, 3, 4):
                if pred.shape[0] <= 8:       # (camera,time,H,W,C)
                    pred = np.transpose(pred, (1, 0, 2, 3, 4))
                elif pred.shape[1] <= 8:     # (time,camera,H,W,C)
                    pass
                else:
                    raise ValueError(f"Unrecognized prediction array shape {pred.shape} in {npz_path}")
            elif pred.ndim == 4 and pred.shape[-1] in (1, 3, 4):
                pred = pred[:, None, ...]
            else:
                raise ValueError(f"Unsupported prediction array shape {pred.shape} in {npz_path}")

            interaction = int(np.asarray(z.get("interaction_index", 0)).item())
            start = int(np.asarray(z.get("start_frame_index", 0)).item())
            for t in range(pred.shape[0]):
                views = []
                for cam in range(pred.shape[1]):
                    im = pred[t, cam]
                    if im.dtype != np.uint8:
                        if np.issubdtype(im.dtype, np.floating) and im.max() <= 1.0:
                            im = im * 255.0
                        im = np.clip(im, 0, 255).astype(np.uint8)
                    if im.shape[-1] == 1:
                        im = np.repeat(im, 3, axis=-1)
                    elif im.shape[-1] == 4:
                        im = im[..., :3]
                    views.append(im)
                # Tile all camera views horizontally, retaining multi-view evidence.
                height = min(v.shape[0] for v in views)
                views = [np.asarray(Image.fromarray(v).resize(
                    (max(1, int(v.shape[1] * height / v.shape[0])), height)
                )) for v in views]
                from PIL import Image as PILImage
                canvas = PILImage.new("RGB", (sum(v.shape[1] for v in views), height))
                x = 0
                for v in views:
                    canvas.paste(PILImage.fromarray(v), (x, 0))
                    x += v.shape[1]
                scale = min(1.0, max_side / max(canvas.size))
                if scale < 1:
                    canvas = canvas.resize((max(1, int(canvas.width*scale)),
                                            max(1, int(canvas.height*scale))))
                frames.append(canvas)
                frame_ids.append(start + t)
    if not frames:
        raise RuntimeError(f"No prediction frames found for {row['episode_id']}")
    # Remove exact duplicate boundary frames if adjacent interactions overlap.
    dedup_frames, dedup_ids = [], []
    seen = set()
    for im, fid in zip(frames, frame_ids):
        if fid in seen:
            continue
        seen.add(fid)
        dedup_frames.append(im)
        dedup_ids.append(fid)
    return dedup_frames, dedup_ids


def extract_json(text):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise ValueError(f"VLM did not return JSON: {text[:400]}")
        return json.loads(m.group(0))


def judge(model, processor, args, row):
    all_frames, all_ids = npz_frames_for_trajectory(row, args.max_side)
    picks = np.linspace(0, len(all_frames)-1, min(args.frames, len(all_frames)), dtype=int)
    images = [all_frames[i] for i in sorted(set(picks))]
    ids = [all_ids[i] for i in sorted(set(picks))]
    content = [{"type": "text", "text":
        f"Task instruction: {row['instruction']}\n"
        f"DROID trajectory ID: {row['episode_id']}\n"
        f"These {len(images)} images are sampled chronologically from the generated "
        f"Ctrl-World replay. Interaction/frame positions: {ids}. Judge the generated prediction."}]
    content.extend({"type": "image", "image": im} for im in images)
    messages = [
        {"role": "system", "content": PROMPT},
        {"role": "user", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    inputs.pop("token_type_ids", None)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)[0]
    result = extract_json(answer)
    verdict = str(result.get("verdict", "uncertain")).lower().strip()
    if verdict not in {"success", "failure", "uncertain"}:
        verdict = "uncertain"
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    if verdict != "uncertain" and confidence < args.min_confidence:
        verdict = "uncertain"
    return {
        "verdict": verdict, "confidence": confidence,
        "evidence": str(result.get("evidence", "")),
        "failure_reason": str(result.get("failure_reason", "")),
        "sampled_frames": len(images), "error": None,
    }


def init_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE IF NOT EXISTS evaluations (
        episode_id TEXT PRIMARY KEY, task_name TEXT, instruction TEXT,
        annotation_path TEXT, gt_success INTEGER, npz_signature TEXT,
        model TEXT, verdict TEXT, confidence REAL, evidence TEXT,
        failure_reason TEXT, sampled_frames INTEGER, error TEXT,
        evaluated_at TEXT DEFAULT CURRENT_TIMESTAMP)""")
    con.commit()
    con.close()


def signature(row):
    h = hashlib.sha256()
    for f in row["npz_files"]:
        st = os.stat(f)
        h.update(f"{f}:{st.st_mtime_ns}:{st.st_size}".encode())
    h.update(str(row["instruction"]).encode())
    h.update(str(row["gt_success"]).encode())
    return h.hexdigest()


def is_done(db, row, model):
    con = sqlite3.connect(db)
    old = con.execute("SELECT npz_signature,model,verdict,error FROM evaluations WHERE episode_id=?",
                      (row["episode_id"],)).fetchone()
    con.close()
    return bool(old and old[0] == signature(row) and old[1] == model
                and old[2] in ("success", "failure", "uncertain") and not old[3])


def save(db, row, model, result):
    con = sqlite3.connect(db, timeout=60)
    con.execute("""INSERT INTO evaluations
        (episode_id,task_name,instruction,annotation_path,gt_success,npz_signature,
         model,verdict,confidence,evidence,failure_reason,sampled_frames,error,evaluated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
        ON CONFLICT(episode_id) DO UPDATE SET
        task_name=excluded.task_name,instruction=excluded.instruction,
        annotation_path=excluded.annotation_path,gt_success=excluded.gt_success,
        npz_signature=excluded.npz_signature,model=excluded.model,
        verdict=excluded.verdict,confidence=excluded.confidence,
        evidence=excluded.evidence,failure_reason=excluded.failure_reason,
        sampled_frames=excluded.sampled_frames,error=excluded.error,
        evaluated_at=CURRENT_TIMESTAMP""",
        (row["episode_id"], row["task_name"], row["instruction"], row["annotation_path"],
         row["gt_success"], signature(row), model, result["verdict"], result["confidence"],
         result["evidence"], result["failure_reason"], result["sampled_frames"], result["error"]))
    con.commit()
    con.close()


def metrics(rows):
    labeled = [r for r in rows if r["gt_success"] in (0, 1)
               and r["verdict"] in ("success", "failure", "uncertain")]
    tp = sum(r["gt_success"] == 1 and r["verdict"] == "success" for r in labeled)
    tn = sum(r["gt_success"] == 0 and r["verdict"] == "failure" for r in labeled)
    fp = sum(r["gt_success"] == 0 and r["verdict"] == "success" for r in labeled)
    fn = sum(r["gt_success"] == 1 and r["verdict"] == "failure" for r in labeled)
    uncertain = sum(r["verdict"] == "uncertain" for r in labeled)
    decided = tp + tn + fp + fn
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
    return {
        "n_droid_labeled": len(labeled), "n_decided": decided, "n_uncertain": uncertain,
        "confusion_matrix": {"TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "accuracy_decided_percent": 100*(tp+tn)/decided if decided else None,
        "precision_success_percent": 100*precision if precision is not None else None,
        "recall_success_percent": 100*recall if recall is not None else None,
        "f1_success_percent": 100*f1 if f1 is not None else None,
        "false_success_rate_percent": 100*fp/(fp+tn) if fp+tn else None,
        "uncertain_rate_percent": 100*uncertain/len(labeled) if labeled else None,
    }


def export(db, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = [dict(x) for x in con.execute("SELECT * FROM evaluations ORDER BY episode_id")]
    con.close()
    fields = ["episode_id", "task_name", "instruction", "verdict", "gt_success",
              "annotation_path", "confidence", "evidence", "failure_reason",
              "sampled_frames", "error", "evaluated_at"]
    with (out/"episode_results.csv").open("w", newline="", encoding="utf-8") as f:
        # Ignore internal SQLite columns (model, npz_signature) in the user-facing CSV.
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    valid = [r for r in rows if r["verdict"] in ("success", "failure", "uncertain")]
    s = sum(r["verdict"] == "success" for r in valid)
    fail = sum(r["verdict"] == "failure" for r in valid)
    u = sum(r["verdict"] == "uncertain" for r in valid)
    n = len(valid)
    summary = {
        "n_evaluated": n, "vlm_success": s, "vlm_failure": fail, "vlm_uncertain": u,
        "vlm_estimated_tcr_success_over_all_percent": 100*s/n if n else None,
        "droid_label_comparison": metrics(rows),
        "per_task": {},
        "warning": "DROID labels validate only the corresponding original/replayed DROID trajectory. They are not ground truth for different newly imagined rollouts."
    }
    for task in sorted({r["task_name"] for r in valid}):
        subset = [r for r in valid if r["task_name"] == task]
        ss = sum(r["verdict"] == "success" for r in subset)
        summary["per_task"][task] = {
            "n": len(subset), "vlm_success": ss,
            "vlm_tcr_percent": 100*ss/len(subset) if subset else None,
            "droid_label_comparison": metrics(subset)
        }
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Evaluated {n} trajectories; VLM success={s}, failure={fail}, uncertain={u}")
    print("DROID comparison:", json.dumps(summary["droid_label_comparison"], indent=2))
    print("Results:", out.resolve())
    print("Resume database:", Path(db).resolve())


def main():
    args = parse_args()
    rows = discover_trajectories(args.rollout_dir, args.dataset_root, args.annotation_split)
    if not rows:
        raise SystemExit("No trajectory NPZ files found. Expected evaluation_data/trajectory_*/interaction_*.npz")
    labeled = sum(r["gt_success"] in (0, 1) for r in rows)
    print(f"Discovered {len(rows)} trajectories; {labeled} have DROID success labels.")
    for r in rows:
        if r["gt_success"] is None:
            print(f"[WARN] No DROID success label for trajectory {r['episode_id']} ({r['annotation_path']})")

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    init_db(args.db)
    print("Loading local VLM:", args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True, local_files_only=args.local_files_only)
    if not torch.cuda.is_available():
        model.to("cpu")
        print("[WARNING] CPU inference will be slow.")
    model.eval()

    todo = [r for r in rows if args.force or not is_done(args.db, r, args.model)]
    print(f"Already completed: {len(rows)-len(todo)}; remaining: {len(todo)}")
    for i, row in enumerate(todo, 1):
        try:
            result = judge(model, processor, args, row)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = {"verdict":"error","confidence":None,"evidence":"",
                      "failure_reason":"","sampled_frames":0,
                      "error":"CUDA OOM; reduce --frames or --max-side."}
        except Exception as e:
            result = {"verdict":"error","confidence":None,"evidence":"",
                      "failure_reason":"","sampled_frames":0,
                      "error":f"{type(e).__name__}: {e}"}
        save(args.db, row, args.model, result)
        print(f"[{i}/{len(todo)}] trajectory {row['episode_id']}: {result['verdict']}", flush=True)
        export(args.db, args.out_dir)
    export(args.db, args.out_dir)


if __name__ == "__main__":
    main()
