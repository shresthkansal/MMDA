"""
Phase 1: builds the reference action library (action_library.json) by running
each reference-take step's video snippet + transcript through a Qwen VLM and
embedding the resulting description.

Ported from Data_Preparation.ipynb (cell 21, the structurally live/self-
triggering Phase 1 version -- cells 10/11/13/15/17/19 are earlier Gemini-
backed or non-Excel iterations, superseded). Verified for real against
Take 3 on Colab: with the existing gold action_library.json (56/56 entries
already "complete"), a full run correctly resumed and skipped every step
with zero real Qwen/Gemini API calls, left the file byte-for-byte unchanged,
and Excel export worked.

Deliberately named apart from action_library.py: that module builds a
differently-shaped action_master_dictionary.json (Whisper transcript +
geometric zone heuristics), flagged in its own docstring as likely
superseded by this Qwen-VLM approach. The two are unrelated artifacts that
happen to share "action library" as a name in casual usage.

Not auto-chained into run.py -- like Phase 0, this only ever runs once
against the reference take, not per student take.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import io
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

from . import config

CAMERA_FILENAMES = ["front.mp4", "side.mp4", "back.mp4"]
TRANSCRIPT_FILENAME = "transcript.txt"

MAX_FRAMES_VLM = 6
TILE_TARGET_H = 360
NORM_WARN_THRESH = 0.95

VLM_MODEL = "qwen3-vl-plus"
QWEN_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

VLM_SYSTEM_PROMPT = (
    "You are an expert clinical examiner documenting a cardiovascular examination reference recording. "
    "Analyze the provided video frames and transcript for the specified examination step. "
    "You must respond with a valid JSON object containing exactly these keys: \n\n"
    "1. \"has_target_action\": boolean (true if the expected clinical action for this step is visible/audible, false otherwise)\n"
    "2. \"visual_evidence\": string (Describe precisely what the doctor is doing. Include body position, hand placement, anatomical site, instrument use [e.g., bell vs diaphragm], and patient position changes)\n"
    "3. \"transcript_alignment\": string (Explain how the verbal narration aligns with the physical action being performed)\n"
    "4. \"temporal_grounding\": string (Identify which part of the action is most distinct based on the frames/transcript)\n"
    "5. \"confidence_score\": float (A number between 0.0 and 1.0 indicating your confidence in this assessment)\n\n"
    "Be highly specific and use precise clinical terminology. Do NOT output markdown code blocks, just the raw JSON."
)


# ==========================================
# JSON / CSV helpers
# ==========================================
def load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def load_timestamps_csv(path: str) -> dict:
    ts_dict = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            full_step_id = f"step_{row['step_id']}_{row['name']}"
            ts_dict[full_step_id] = {
                "start_sec": float(row["start_time"]),
                "end_sec": float(row["end_time"]),
            }
    return ts_dict


# ==========================================
# Frame extraction + tiling
# ==========================================
def _extract_frames_from_snippet(video_path: str, max_n: int = MAX_FRAMES_VLM) -> List[np.ndarray]:
    vid_name = os.path.basename(video_path)
    if not os.path.exists(video_path):
        tqdm.write(f"      [{vid_name}] ERROR: File missing: {video_path}")
        return []

    fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        shutil.copy2(video_path, temp_path)
    except Exception as e:
        tqdm.write(f"      [{vid_name}] ERROR: Failed to copy locally: {e}")
        return []

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return []

    indices = [int(i * (total_frames - 1) / (max_n - 1)) for i in range(max_n)] if max_n > 1 else [total_frames // 2]

    frames: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx, total_frames - 1))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
    if os.path.exists(temp_path):
        os.remove(temp_path)
    return frames


async def extract_snippet_camera_frames(video_paths: List[str]) -> List[List[np.ndarray]]:
    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, _extract_frames_from_snippet, vp) for vp in video_paths]
    return list(await asyncio.gather(*tasks))


def tile_frame_groups(
    frame_groups: List[List[np.ndarray]],
    target_h: int = TILE_TARGET_H,
    max_width: int = 1920,
) -> List[Image.Image]:
    n = max((len(g) for g in frame_groups), default=0)
    if n == 0:
        return []

    tiled: List[Image.Image] = []
    for i in range(n):
        panels: List[Image.Image] = []
        for group in frame_groups:
            arr = (group[i] if i < len(group) else group[-1]) if group else \
                np.zeros((target_h, target_h * 4 // 3, 3), dtype=np.uint8)
            img = Image.fromarray(arr)
            new_w = max(1, int(img.width * target_h / img.height))
            panels.append(img.resize((new_w, target_h), Image.LANCZOS))

        total_w = sum(p.width for p in panels)
        canvas = Image.new("RGB", (total_w, target_h))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0))
            x += panel.width

        if canvas.width > max_width:
            canvas = canvas.resize((max_width, int(canvas.height * max_width / canvas.width)), Image.LANCZOS)
        tiled.append(canvas)
    return tiled


def subsample_evenly(frames: List[Image.Image], max_n: int) -> List[Image.Image]:
    if len(frames) <= max_n or max_n <= 0:
        return frames
    if max_n == 1:
        return [frames[len(frames) // 2]]
    indices = [round(i * (len(frames) - 1) / (max_n - 1)) for i in range(max_n)]
    return [frames[idx] for idx in indices]


# ==========================================
# VLM + embedding
# ==========================================
async def vlm_describe_step(
    qwen_client,
    frames: List[Image.Image],
    transcript_text: str,
    step_id: str,
    semaphore: asyncio.Semaphore,
) -> str:
    def pil_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    prompt_text = (
        f"{VLM_SYSTEM_PROMPT}\n\n"
        f"Examination step: {step_id}\n"
        f"Frames: {len(frames)} (evenly sampled from snippet; each shows Front | Side | Back° cameras side-by-side)\n\n"
        f"Step Transcript:\n{transcript_text or '[no speech detected in this step]'}\n\n"
    )

    max_retries = 5
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await qwen_client.chat.completions.create(
                    model=VLM_MODEL,
                    messages=[
                        {"role": "system", "content": VLM_SYSTEM_PROMPT},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt_text},
                            *[{"type": "image_url", "image_url": {"url": pil_to_data_url(img)}} for img in frames],
                        ]},
                    ],
                    max_tokens=800,
                    temperature=0.2,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    if attempt < max_retries - 1:
                        wait_time = 20 * (attempt + 1)
                        tqdm.write(f"      [Rate Limit] Qwen quota hit. Retrying in {wait_time}s... "
                                   f"(Attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(wait_time)
                    else:
                        raise Exception(f"Max retries exceeded for VLM on step {step_id}") from e
                else:
                    raise e


_st_cache: Dict[str, Any] = {}


def _st_embed_sync(model_name: str, text: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    if model_name not in _st_cache:
        _st_cache[model_name] = SentenceTransformer(model_name)
    return _st_cache[model_name].encode(text, normalize_embeddings=True).astype(np.float32)


async def embed_description(
    async_client,
    text: str,
    backend: str,
    semaphore: asyncio.Semaphore,
) -> List[float]:
    if backend.startswith("gemini:"):
        model = backend.split(":", 1)[1]
        async with semaphore:
            resp = await async_client.models.embed_content(model=model, contents=text)
        vec = np.array(resp.embeddings[0].values, dtype=np.float32)
    elif backend.startswith("sentence-transformers:"):
        model_name = backend.split(":", 1)[1]
        loop = asyncio.get_running_loop()
        vec = await loop.run_in_executor(None, _st_embed_sync, model_name, text)
    else:
        raise ValueError(f"Unsupported embed_backend: {backend!r}")

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# ==========================================
# Config parsing
# ==========================================
def parse_steps(phases_cfg: dict) -> List[Tuple[str, str, str]]:
    steps: List[Tuple[str, str, str]] = []
    if "umbrellas" in phases_cfg:
        for u in phases_cfg["umbrellas"]:
            uid, uname = u.get("id", ""), u.get("name", u.get("id", ""))
            for s in u.get("steps", []):
                sid = s.get("step_id") or s.get("id") or s.get("name", "")
                if sid:
                    steps.append((sid, uid, uname))
    return steps


def find_co_located_group(step_id: str, phases_cfg: dict) -> Optional[List[str]]:
    for group in phases_cfg.get("co_located_groups", []):
        if isinstance(group, list) and step_id in group:
            return group
    return None


def step_is_splittable(step_id: str, phases_cfg: dict) -> bool:
    return any(isinstance(entry, str) and entry == step_id for entry in phases_cfg.get("splittable_steps", []))


# ==========================================
# Per-step pipeline
# ==========================================
async def process_one_step(
    step_id: str,
    umbrella_id: str,
    umbrella_name: str,
    timestamps: dict,
    snippets_root: str,
    embed_client,
    qwen_client,
    embed_backend: str,
    vlm_sem: asyncio.Semaphore,
    emb_sem: asyncio.Semaphore,
    phases_cfg: dict,
) -> dict:
    tqdm.write(f"\n▶ [{step_id}] STARTED PROCESSING...")
    ts = timestamps.get(step_id, {"start_sec": 0.0, "end_sec": 0.0})

    step_dir = os.path.join(snippets_root, step_id)
    if not os.path.exists(step_dir):
        tqdm.write(f"  [{step_id}] ❌ ERROR: Folder missing: {step_dir}")
        return {"step_id": step_id, "status": "folder_missing", "description_length": None, "embedding_norm": None}

    tx_path = os.path.join(step_dir, TRANSCRIPT_FILENAME)
    tx_text = ""
    if os.path.exists(tx_path):
        with open(tx_path, "r", encoding="utf-8") as f:
            tx_text = f.read().strip()

    video_paths = [os.path.join(step_dir, fname) for fname in CAMERA_FILENAMES]
    frame_groups = await extract_snippet_camera_frames(video_paths)
    tiled = tile_frame_groups(frame_groups)
    tiled = subsample_evenly(tiled, MAX_FRAMES_VLM)

    if tiled:
        description = await vlm_describe_step(qwen_client, tiled, tx_text, step_id, vlm_sem)
    else:
        description = "{}"
        tqdm.write(f"  [{step_id}]    -> Skipped Qwen (no frames found)")

    try:
        vlm_json = json.loads(description)
        text_to_embed = f"{vlm_json.get('visual_evidence', '')} {vlm_json.get('transcript_alignment', '')}".strip()
        if not text_to_embed:
            text_to_embed = "[No visual or transcript description found]"
    except json.JSONDecodeError:
        tqdm.write(f"  [{step_id}] ⚠️ WARN: VLM did not return valid JSON. Falling back.")
        vlm_json = {"raw_text": description, "has_target_action": False, "confidence_score": 0.0}
        text_to_embed = description

    embedding = await embed_description(embed_client, text_to_embed, embed_backend, emb_sem)
    norm = float(np.linalg.norm(np.array(embedding, dtype=np.float32)))

    record = {
        "step_id": step_id, "umbrella": umbrella_id, "umbrella_name": umbrella_name,
        "description": vlm_json, "embedding": embedding,
        "source_timestamps": {"start_sec": ts["start_sec"], "end_sec": ts["end_sec"]},
        "co_located_group": find_co_located_group(step_id, phases_cfg),
        "is_splittable": step_is_splittable(step_id, phases_cfg),
        "vlm_model": VLM_MODEL,
    }
    tqdm.write(f"✔ [{step_id}] FINISHED SUCCESSFULLY.")
    return {
        "_record": record, "step_id": step_id, "description_length": len(text_to_embed),
        "embedding_norm": norm, "status": "ok" if norm >= NORM_WARN_THRESH else "norm_warn",
    }


_COL = [52, 20, 16, 14]
_SEP = "─" * (sum(_COL) + len(_COL) * 3)


def print_progress_table(rows: List[dict]) -> None:
    header = (f"{'step_id':<{_COL[0]}}   {'description_length':>{_COL[1]}}   "
              f"{'embedding_norm':>{_COL[2]}}   {'status':>{_COL[3]}}")
    print(f"\n{header}")
    print(_SEP)
    for r in rows:
        sid = r["step_id"][: _COL[0]]
        dl = str(r["description_length"]) if r["description_length"] is not None else "—"
        en = f"{r['embedding_norm']:.5f}" if r["embedding_norm"] is not None else "—"
        print(f"{sid:<{_COL[0]}}   {dl:>{_COL[1]}}   {en:>{_COL[2]}}   {r.get('status', ''):>{_COL[3]}}")
    print(_SEP)


def export_to_excel(json_path: str, excel_path: str) -> None:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]

    rows = []
    for item in data:
        desc = item.get("description", {}) if isinstance(item.get("description"), dict) else {}
        ts = item.get("source_timestamps", {}) if isinstance(item.get("source_timestamps"), dict) else {}
        co_located = item.get("co_located_group")
        co_located_str = (", ".join(str(x) for x in co_located) if isinstance(co_located, list)
                           else (str(co_located) if co_located else None))
        rows.append({
            "step_id": item.get("step_id"), "umbrella": item.get("umbrella"),
            "umbrella_name": item.get("umbrella_name"),
            "has_target_action": item.get("has_target_action", desc.get("has_target_action")),
            "visual_evidence": item.get("visual_evidence", desc.get("visual_evidence")),
            "transcript_alignment": item.get("transcript_alignment", desc.get("transcript_alignment")),
            "temporal_grounding": item.get("temporal_grounding", desc.get("temporal_grounding")),
            "confidence_score": item.get("confidence_score", desc.get("confidence_score")),
            "start_sec": item.get("start_sec", ts.get("start_sec")),
            "end_sec": item.get("end_sec", ts.get("end_sec")),
            "co_located_group": co_located_str, "is_splittable": item.get("is_splittable"),
            "vlm_model": item.get("vlm_model"),
        })

    df = pd.DataFrame(rows)
    target_columns = ["step_id", "umbrella", "umbrella_name", "has_target_action", "visual_evidence",
                       "transcript_alignment", "temporal_grounding", "confidence_score", "start_sec",
                       "end_sec", "co_located_group", "is_splittable", "vlm_model"]
    df = df[[c for c in target_columns if c in df.columns]]
    df.to_excel(excel_path, index=False)
    print(f"Successfully exported {len(df)} rows to {excel_path}")


# ==========================================
# Orchestrator
# ==========================================
async def build_reference_library(
    take_id: int,
    step_concurrency: int = 1,
    vlm_concurrency: int = 1,
    embed_concurrency: int = 1,
) -> List[dict]:
    """Runs every step in phases_config.json's taxonomy through the Qwen VLM +
    embedding pipeline, resumable (skips steps whose cached description is
    already > 400 chars), and writes the updated action_library.json + Excel
    export to the take's processed folder. Returns the progress rows."""
    from google import genai
    from openai import AsyncOpenAI

    p = config.get_take_paths(take_id)

    if not os.path.exists(p["phases_config"]):
        raise FileNotFoundError(f"Missing phases_config.json at: {p['phases_config']} — run phase0.run_phase0() first.")
    if not os.path.exists(p["annotations_csv"]):
        raise FileNotFoundError(f"Missing annotations CSV at: {p['annotations_csv']}")

    phases_cfg = load_json(p["phases_config"])
    timestamps = load_timestamps_csv(p["annotations_csv"])
    embed_backend: str = phases_cfg["provenance"]["embed_backend"]

    existing_all: Dict[str, dict] = {}
    existing_long: Dict[str, dict] = {}
    if os.path.exists(p["reference_library_json"]):
        try:
            for entry in load_json(p["reference_library_json"]):
                sid = entry.get("step_id")
                if not sid:
                    continue
                existing_all[sid] = entry
                if len(str(entry.get("description", ""))) > 400:
                    existing_long[sid] = entry
        except Exception:
            pass

    steps = parse_steps(phases_cfg)
    print(f"{len(steps)} total steps  |  {len(steps) - len(existing_long)} to process  |  {len(existing_long)} skipped")

    embed_client = genai.Client().aio
    qwen_client = AsyncOpenAI(api_key=os.environ["QWEN_API_KEY"], base_url=QWEN_BASE_URL)

    step_sem = asyncio.Semaphore(step_concurrency)
    vlm_sem = asyncio.Semaphore(vlm_concurrency)
    emb_sem = asyncio.Semaphore(embed_concurrency)
    lib_lock = asyncio.Lock()

    progress_rows: List[dict] = []

    for sid, uid, uname in steps:
        if sid in existing_long:
            entry = existing_long[sid]
            emb = entry.get("embedding", [])
            desc_val = entry.get("description", "")
            desc_len = (len(str(desc_val.get("visual_evidence", ""))) if isinstance(desc_val, dict)
                        else len(str(desc_val)))
            progress_rows.append({
                "step_id": sid, "description_length": desc_len,
                "embedding_norm": float(np.linalg.norm(np.array(emb, dtype=np.float32))) if emb else None,
                "status": "skipped",
            })

    to_do = [s for s in steps if s[0] not in existing_long]
    pbar = tqdm(total=len(to_do), desc="Processing snippets", unit="step")

    async def handle(step_id: str, uid: str, uname: str) -> None:
        if step_id in existing_long:
            return
        async with step_sem:
            try:
                result = await process_one_step(
                    step_id, uid, uname, timestamps, config.ACTION_DICTIONARY_ROOT,
                    embed_client, qwen_client, embed_backend, vlm_sem, emb_sem, phases_cfg,
                )
            except Exception as exc:
                tqdm.write(f"  ERROR  {step_id}: {type(exc).__name__}: {exc}")
                progress_rows.append({"step_id": step_id, "description_length": None,
                                       "embedding_norm": None, "status": f"error: {type(exc).__name__}"})
                pbar.update(1)
                return

            if "_record" in result:
                async with lib_lock:
                    existing_all[step_id] = result["_record"]
                    save_json(p["reference_library_json"], list(existing_all.values()))

            progress_rows.append({k: v for k, v in result.items() if k != "_record"})
            pbar.update(1)
            await asyncio.sleep(3)

    await asyncio.gather(*(handle(sid, uid, uname) for sid, uid, uname in steps))
    pbar.close()

    order = {sid: i for i, (sid, *_rest) in enumerate(steps)}
    progress_rows.sort(key=lambda r: order.get(r["step_id"], 9999))
    print_progress_table(progress_rows)

    print(f"Action library saved to:\n  {p['reference_library_json']}")
    export_to_excel(p["reference_library_json"], p["reference_library_excel"])

    return progress_rows


def run_reference_library(take_id: int, **kwargs) -> List[dict]:
    """Sync entry point -- wraps build_reference_library() in asyncio.run()."""
    return asyncio.run(build_reference_library(take_id, **kwargs))
