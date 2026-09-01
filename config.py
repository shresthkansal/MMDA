"""
Environment, path, and per-take configuration.

Ported from FS_model.ipynb (Section 2: Configuration & Path Setup, cell idx 8).
This is the single source of truth for take file locations and sync offsets --
do not duplicate TAKE_CONFIG/SYNC_CONFIG elsewhere.
"""
import os
import platform

import torch

# ==========================================
# 1. GLOBAL CONFIGURATION (DATA & SYNC)
# ==========================================

RAW_DATA_ROOT = "/content/drive/MyDrive/Raw_Data"
PROCESSED_DATA_ROOT = "/content/drive/MyDrive/Processed_Data"

# Shared reference-snippet library used by Phase 1 (reference_library.py) -- not
# per-take, one directory of per-step snippet folders for the reference take.
ACTION_DICTIONARY_ROOT = os.path.join(PROCESSED_DATA_ROOT, "dataset/action_dictionary")

TAKE_CONFIG = {
    2: {
        "front": ["0004SQ", "00056X"],
        "side": ["0004VM", "000558"],
        "360": ["00053D.mp4"],
    },
    3: {
        "front": ["00062L", "0007UD"],
        "side": ["0006OB", "0007JK"],
        "360": ["00073D.mp4"],
    },
    5: {
        "front": ["00092I", "0010XD"],
        "side": ["0009ML", "0010LQ"],
        "360": ["00103D.mp4"],
    },
    6: {
        "front": ["00116E", "0012C5"],
        "side": ["0011H6", "0012EP"],
        "360": ["00123D.mp4"],
    },
}

SYNC_CONFIG = {
    2: {
        "offset_delay_seconds": 3.64263,
        "start_trim_seconds": 27.0,
        "audio_source_camera": "side",
        "audio_track_L": "00",
        "audio_track_R": "01",
        "360_offset_seconds": 17.10073,
    },
    3: {
        "offset_delay_seconds": 18.37805,
        "start_trim_seconds": 35.89020,
        "audio_source_camera": "side",
        "audio_track_L": "00",
        "audio_track_R": "01",
        "360_offset_seconds": 35.89020,
    },
    5: {
        "offset_delay_seconds": -9.09075,
        "start_trim_seconds": 10.0,
        "audio_source_camera": "side",
        "audio_track_L": "00",
        "audio_track_R": "01",
        "360_offset_seconds": -14.87615,
    },
    6: {
        "offset_delay_seconds": -3.80776,
        "start_trim_seconds": 5.0,
        "audio_source_camera": "side",
        "audio_track_L": "00",
        "audio_track_R": "01",
        "360_offset_seconds": -5.49950,
    },
}


# ==========================================
# 2. ENVIRONMENT & HELPERS
# ==========================================
def mount_drive() -> None:
    """Mount Google Drive if running in Colab and not already mounted."""
    if not os.path.exists("/content/drive"):
        from google.colab import drive
        drive.mount("/content/drive")
    else:
        print("Google Drive is already mounted.")


def get_environment_config() -> str:
    """Detects hardware automatically: 'macos', 'colab' (has CUDA), or 'cpu'."""
    if platform.system() == "Darwin":
        return "macos"
    elif torch.cuda.is_available():
        return "colab"
    else:
        return "cpu"


def get_take_files(take_id: int) -> dict:
    if take_id not in TAKE_CONFIG:
        raise ValueError(f"Take ID {take_id} not found in TAKE_CONFIG")
    return TAKE_CONFIG[take_id]


def get_sync_config(take_id: int) -> dict:
    if take_id not in SYNC_CONFIG:
        raise ValueError(f"Sync config for Take ID {take_id} not found in SYNC_CONFIG")
    return SYNC_CONFIG[take_id]


def resolve_media_path(raw_folder: str, name: str, fallback_ext: str = ".MXF") -> "str | None":
    """Finds a file in the raw folder, handling case-sensitivity and extensions."""
    if not name:
        return None

    if os.path.splitext(name)[1]:
        candidate = os.path.join(raw_folder, name)
        if os.path.exists(candidate):
            return candidate

    for ext in [fallback_ext, fallback_ext.lower()]:
        candidate = os.path.join(raw_folder, f"{name}{ext}")
        if os.path.exists(candidate):
            return candidate

    # Return prediction even if missing (for logging)
    return os.path.join(raw_folder, f"{name}{fallback_ext}")


# ==========================================
# 3. MASTER PATH BUILDER
# ==========================================
def _first_existing(*candidates: str) -> str:
    """First path that exists, else the first candidate (so callers still get
    a sensible path to create or to name in an error message)."""
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


def get_take_paths(take_id: int) -> dict:
    """Generates all file paths for a given Take ID, creating directories as needed."""
    raw_root = os.path.join(RAW_DATA_ROOT, f"Take_{take_id}")
    proc_root = os.path.join(PROCESSED_DATA_ROOT, f"Take_{take_id}")
    pose_root = os.path.join(proc_root, "Pose_Results_YOLO")
    reid_root = os.path.join(proc_root, "ReID_Results")

    for folder in [raw_root, proc_root, pose_root, reid_root]:
        os.makedirs(folder, exist_ok=True)

    return {
        "id": take_id,
        "raw_folder": raw_root,
        "processed_folder": proc_root,
        "pose_folder": pose_root,
        "reid_folder": reid_root,

        # --- Synced video outputs ---
        "vid_side": os.path.join(proc_root, "Final_Side_Muxed.mp4"),
        "vid_front": os.path.join(proc_root, "Final_Front_Muxed.mp4"),
        "vid_360": os.path.join(proc_root, "Final_360_Muxed.mp4"),
        "vid_verify": os.path.join(proc_root, "Verification_Preview.mp4"),
        "audio": os.path.join(proc_root, "Final_Audio.aac"),

        # --- Raw per-view YOLO keypoint CSVs (output of pose.run_pose_extraction_batched) ---
        "pose_side_csv": os.path.join(pose_root, "Side_Keypoints.csv"),
        "pose_front_csv": os.path.join(pose_root, "Front_Keypoints.csv"),
        "pose_360_csv": os.path.join(pose_root, "360_Keypoints.csv"),

        # --- Labeled pose outputs (post role-assignment) ---
        "pose_side_labeled_csv": os.path.join(reid_root, "Side_Final.csv"),
        "pose_front_labeled_csv": os.path.join(reid_root, "Front_Final.csv"),
        "pose_360_labeled_csv": os.path.join(reid_root, "360_Final.csv"),

        # --- Wide keypoints dataset ---
        "wide_keypoints_csv": os.path.join(proc_root, f"Take_{take_id}_Wide_Keypoints2.csv"),

        # --- Transcript ---
        "transcript": os.path.join(proc_root, "Transcript.txt"),

        # --- Phase 0 outputs (phase0.py) ---
        "phases_config": os.path.join(proc_root, "phases_config.json"),
        "anchor_embeddings": os.path.join(proc_root, "anchor_embeddings.npy"),
        "anchor_index": os.path.join(proc_root, "anchor_index.json"),
        "phase0_validation_log": os.path.join(proc_root, "phase0_validation.log"),

        # --- Phase 1 outputs (reference_library.py) ---
        # Ground-truth annotations are not named consistently across takes:
        # Take 3 has annotations_master.csv, Take 2 only has
        # manual_annotations_take2.csv. Resolve to whichever exists (the
        # master name wins if both do, as on Take 3). The two files also use
        # different column schemas -- segmentation.read_annotations()
        # normalises that, see its docstring.
        "annotations_csv": _first_existing(
            os.path.join(proc_root, "annotations_master.csv"),
            os.path.join(proc_root, f"manual_annotations_take{take_id}.csv"),
        ),
        "reference_library_json": os.path.join(proc_root, "action_library.json"),
        "reference_library_excel": os.path.join(proc_root, "action_library.xlsx"),

        # --- Stage 01 segmentation outputs (segmentation.py) ---
        # NB: deliberately NOT "segments.json". Take_2/segments.json already
        # holds unrelated prior work (6 umbrella-level U1-U6 spans with VLM
        # evidence text, from the Phase 2 / cvs_pipeline2 line), which this
        # module would otherwise silently overwrite on its first run.
        "segmentation_log": os.path.join(proc_root, "segmentation.log"),
        "segments_json": os.path.join(proc_root, "stage01_segments.json"),
    }
