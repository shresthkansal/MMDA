"""
Audio sync calculation and audio auditing.

Ported from FS_model.ipynb (Section 3: Audio Analysis & Sync Calculation,
cells idx 11-12). Confirmed working via real run evidence (idx 23, 25 --
computed real offsets for Take 2 and Take 6).

NOTE: the original notebook referenced a global `TEMP_DIR` that was never
actually defined in any cell (relied on stray interactive session state).
Defined properly here instead.
"""
import os
import subprocess
from typing import List, Optional, Tuple

import librosa
import numpy as np
from scipy import signal

TEMP_DIR = "/content/osce_pipeline_tmp"


# ==========================================
# Audio audit
# ==========================================
def _build_source_manifest(raw_folder: str, prefix: str, tracks: List[str]) -> List[Tuple[str, str]]:
    """Return tuples of (source_path, label) for either MXF tracks or single files."""
    has_extension = os.path.splitext(prefix)[1] != ""
    if has_extension:
        label = os.path.splitext(os.path.basename(prefix))[0]
        source_path = prefix if os.path.isabs(prefix) else os.path.join(raw_folder, prefix)
        return [(source_path, label)]

    manifest = []
    for track_id in tracks:
        filename = f"{prefix}{track_id}"
        if not track_id.lower().endswith(".mxf"):
            filename += ".MXF"
        manifest.append((os.path.join(raw_folder, filename), track_id))
    return manifest


def _extract_audio_clip(source_path: str, wav_path: str, duration_seconds: int) -> bool:
    cmd = (
        f'ffmpeg -i "{source_path}" -t {duration_seconds} -ac 1 -vn '
        f'"{wav_path}" -y -hide_banner -loglevel error'
    )
    completed = subprocess.run(cmd, shell=True)
    return completed.returncode == 0


def run_audio_audit(raw_folder: str, prefix: str, tracks: Optional[List[str]] = None,
                     duration_seconds: int = 30) -> dict:
    """Calculate RMS stats for MXF tracks or standalone media files (e.g., MP4)."""
    if tracks is None:
        tracks = ["00", "01", "02", "03"]

    os.makedirs(TEMP_DIR, exist_ok=True)
    results = {}
    manifest = _build_source_manifest(raw_folder, prefix, tracks)

    for source_path, label in manifest:
        safe_prefix = os.path.splitext(os.path.basename(prefix))[0]
        wav_file = os.path.join(TEMP_DIR, f"temp_{safe_prefix}_{label}.wav")

        if not os.path.exists(source_path):
            results[label] = {"status": "missing", "rms": None}
            continue

        if not _extract_audio_clip(source_path, wav_file, duration_seconds):
            results[label] = {"status": "error", "rms": None}
            try:
                os.remove(wav_file)
            except OSError:
                pass
            continue

        try:
            y, sr = librosa.load(wav_file, sr=None)
            rms = float(np.sqrt(np.mean(y ** 2)))
            status = "silent" if rms < 0.001 else "active"
            results[label] = {"status": status, "rms": rms}
        except Exception:
            results[label] = {"status": "error", "rms": None}
        finally:
            try:
                os.remove(wav_file)
            except OSError:
                pass

    return results


# ==========================================
# Sync calculator
# ==========================================
def convert_to_wav(source_path: str, temp_wav_path: str, sample_rate: int = 22050) -> Optional[str]:
    if not os.path.exists(source_path):
        print(f"ERROR: Cannot find {source_path}")
        return None
    cmd = (
        f'ffmpeg -i "{source_path}" -ac 1 -ar {sample_rate} -vn '
        f'"{temp_wav_path}" -y -hide_banner -loglevel error'
    )
    completed = subprocess.run(cmd, shell=True)
    if completed.returncode != 0:
        print(f"ffmpeg failed while reading {source_path}")
        return None
    return temp_wav_path


def _resolve_audio_source(raw_folder: str, identifier: str, track_suffix: Optional[str]) -> Optional[str]:
    base, ext = os.path.splitext(identifier)
    if ext:
        candidate = identifier
    else:
        if track_suffix is None:
            raise ValueError(f"Track suffix required for prefix '{identifier}'")
        suffix = track_suffix if track_suffix.lower().endswith(".mxf") else f"{track_suffix}.MXF"
        candidate = f"{identifier}{suffix}"

    full_path = candidate if os.path.isabs(candidate) else os.path.join(raw_folder, candidate)
    if not os.path.exists(full_path):
        print(f"ERROR: Cannot find {full_path}")
        return None
    return full_path


def run_sync_calculator(
    raw_folder: str,
    prefixes: dict,
    camera_a: str = "front",
    camera_b: str = "side",
    track_a: Optional[str] = "00",
    track_b: Optional[str] = "00",
    duration_seconds: int = 60,
) -> Optional[float]:
    """Cross-correlate audio from two cameras to estimate offset (camera_a - camera_b), in seconds."""
    os.makedirs(TEMP_DIR, exist_ok=True)

    if camera_a not in prefixes or camera_b not in prefixes:
        print(f"Camera keys '{camera_a}' or '{camera_b}' missing from prefixes")
        return None

    prefix_a = prefixes[camera_a][0]
    prefix_b = prefixes[camera_b][0]

    try:
        source_a = _resolve_audio_source(raw_folder, prefix_a, track_a)
        source_b = _resolve_audio_source(raw_folder, prefix_b, track_b)
    except ValueError as exc:
        print(str(exc))
        return None

    if not (source_a and source_b):
        return None

    wav_a = convert_to_wav(source_a, os.path.join(TEMP_DIR, f"temp_{camera_a}_scratch.wav"))
    wav_b = convert_to_wav(source_b, os.path.join(TEMP_DIR, f"temp_{camera_b}_scratch.wav"))

    if not (wav_a and wav_b):
        print("Could not convert audio sources. Check paths.")
        return None

    try:
        print(f"Loading audio (first {duration_seconds}s) for {camera_a} vs {camera_b}...")
        y_a, sr = librosa.load(wav_a, duration=duration_seconds, sr=22050)
        y_b, sr = librosa.load(wav_b, duration=duration_seconds, sr=22050)

        print("Running cross-correlation...")
        correlation = signal.correlate(y_a, y_b, mode="full")
        lags = signal.correlation_lags(y_a.size, y_b.size, mode="full")
        lag = lags[np.argmax(correlation)]
        offset_seconds = lag / sr
        return float(offset_seconds)
    finally:
        for temp in (wav_a, wav_b):
            try:
                if temp and os.path.exists(temp):
                    os.remove(temp)
            except OSError:
                pass
