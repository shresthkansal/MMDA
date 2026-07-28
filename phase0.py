"""
Phase 0: taxonomy/anchor definitions, input validation, and anchor-phrase
embedding precomputation for a reference take.

Ported from Data_Preparation.ipynb (cell 8, the only structurally live/
self-triggering Phase 0 version -- cells 4-6 are an earlier abandoned draft,
see the module's own git history). Verified for real against Take 3 on
Colab: taxonomy/file/schema/dependency checks and anchor embeddings all
matched the existing gold phases_config.json/anchor_embeddings.npy/
anchor_index.json exactly (only the deliberately-redirected scratch paths
differed).

Unlike the notebook version, this reuses config.get_take_paths() instead of
a locally redefined path builder -- the notebook's own get_take_paths() had
the same wide_csv ReID_Results/ path bug fixed in config.py.

REQUIRED_KEYPOINT_COLUMNS deliberately excludes the 56 step_* columns and
Audio_Umbrella_Prediction that the notebook's version required: those come
from construct_binary_step_columns (removed from features.py -- was always
all-zero dead code) and merge_llm_features (only run when a Gemini key is
passed), neither of which features.py's output is guaranteed to have.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import config

# ==========================================
# Taxonomy (umbrellas + steps)
# ==========================================
FPS = 24.0

UMBRELLA_ORDER: List[str] = ["U1", "U2", "U3", "U4", "U5", "U6"]

UMBRELLA_DESCRIPTIONS: Dict[str, str] = {
    "U1": "Preparation",
    "U2": "Inspection",
    "U3": "Upper limb peripheral",
    "U4": "Head / neck",
    "U5": "Precordium",
    "U6": "Back / legs / conclusion",
}

UMBRELLA_STEPS: Dict[str, List[str]] = {
    "U1": [
        "step_1_Greet_Patient", "step_2_Obtain_Consent", "step_3_Respect_Privacy",
        "step_4_Explain_Exam", "step_6_Patient_Comfort", "step_7_Position_45_Degrees",
        "step_5_Hand_Hygiene", "step_8_Expose_Chest_Legs",
    ],
    "U2": [
        "step_9_General_Inspection", "step_10_1_Inspect_Syndromic_General",
        "step_11_Inspect_Anterior_Chest", "step_12_Cover_Chest",
        "step_10_2_Inspect_Syndromic_Legs",
    ],
    "U3": [
        "step_13_Inspect_Hands", "step_15_Palpate_Radial_Pulse",
        "step_16_Palpate_Radial_Symmetry", "step_17_Radio_Femoral_Delay",
        "step_14_Inspect_Forearms", "step_18_Collapsing_Pulse",
    ],
    "U4": [
        "step_19_Inspect_Face", "step_20_1_Inspect_Neck_Corrigan",
        "step_21_Auscultate_Carotids", "step_22_Re_expose_Chest",
        "step_20_2_Inspect_JVP", "step_20_3_Abdominojugular_Reflux",
    ],
    "U5": [
        "step_23_1_Palpate_Apex_Location", "step_23_2_Palpate_Apex_Character",
        "step_24_Palpate_Heave",
        "step_25_1_Thrill_Apex", "step_25_2_Thrill_Left_Sternal",
        "step_25_3_Thrill_Pulmonary", "step_25_4_Thrill_Aortic",
        "step_26_1_Palpate_P2", "step_26_2_Palpate_A2",
        "step_27_1_Explain_Auscultation_Steps", "step_28_1_Explain_Carotid_Timing",
        "step_29_1_Explain_Bell_Step",
        "step_27_2_Auscultate_Mitral_Diaphragm", "step_28_2_Time_Carotid_Pulse",
        "step_29_2_Auscultate_Mitral_Bell",
        "step_30_1_Turn_Left_Lateral", "step_30_2_Palpate_Apex_Lateral",
        "step_30_3_Auscultate_Mitral_Bell_Lateral", "step_30_4_Auscultate_Breathing_Maneuver",
        "step_31_1_Turn_Back_Supine", "step_31_2_Auscultate_Tricuspid",
        "step_32_Auscultate_Pulmonary", "step_33_Auscultate_Aortic",
        "step_34_Auscultate_Carotid_Bruits",
        "step_35_1_Sit_Up_Lean_Forward", "step_35_2_Auscultate_Aortic_Regurgitation",
    ],
    "U6": [
        "step_36_Inspect_Back", "step_37_Auscultate_Lung_Bases",
        "step_38_1_Inspect_Legs_Edema", "step_39_Thank_Patient_Cover",
        "step_40_Final_Summary",
    ],
}

CO_LOCATED_GROUPS: List[List[str]] = [
    ["step_1_Greet_Patient", "step_2_Obtain_Consent", "step_3_Respect_Privacy",
     "step_4_Explain_Exam", "step_6_Patient_Comfort", "step_7_Position_45_Degrees"],
    ["step_9_General_Inspection", "step_10_1_Inspect_Syndromic_General"],
    ["step_27_1_Explain_Auscultation_Steps", "step_28_1_Explain_Carotid_Timing",
     "step_29_1_Explain_Bell_Step"],
    ["step_27_2_Auscultate_Mitral_Diaphragm", "step_28_2_Time_Carotid_Pulse"],
]

SPLITTABLE_STEPS: Dict[str, List[str]] = {
    "step_10": ["step_10_1_Inspect_Syndromic_General", "step_10_2_Inspect_Syndromic_Legs"],
    "step_20": ["step_20_1_Inspect_Neck_Corrigan", "step_20_2_Inspect_JVP",
                "step_20_3_Abdominojugular_Reflux"],
    "step_23": ["step_23_1_Palpate_Apex_Location", "step_23_2_Palpate_Apex_Character"],
}

WEAK_SEQUENTIAL_FALLBACK: List[Tuple[str, str]] = [
    ("step_15_Palpate_Radial_Pulse", "step_16_Palpate_Radial_Symmetry"),
    ("step_39_Thank_Patient_Cover", "step_40_Final_Summary"),
]

PIPELINE_CONFIG: Dict = {
    "score_pass": 0.60,
    "score_flag_poor": 0.50,
    "score_flag_missing": 0.20,
    "anchor_sim_high": 0.75,
    "anchor_sim_medium": 0.60,
    "anchor_sim_low": 0.45,
    "boundary_slop_sec": 30,
    "smoothing_window_frames": 5,
    "sample_fps_coarse": 1.0,
    "sample_fps_fine": 2.0,
    "lead_lag_tolerance_sec": 3.0,
    "colocation_inherit_window_sec": 2.0,
    "fps": FPS,
}


@dataclass
class UmbrellaAnchor:
    umbrella: str
    phrases: List[str]
    role: str = "umbrella_start"
    required: bool = False
    confidence: float = 1.0


@dataclass
class StepAnchor:
    step: str
    phrases: List[str]
    role: str
    lead_lag_sec: float = 0.0
    confidence: float = 1.0


UMBRELLA_ANCHORS: List[UmbrellaAnchor] = [
    UmbrellaAnchor("U1", ["how do i address", "hi ", "hello"], confidence=0.8),
    UmbrellaAnchor("U1", ["cardiovascular exam"], confidence=1.0),
    UmbrellaAnchor("U2", ["general inspection", "comfortable at rest",
                           "no signs of respiratory"], confidence=1.0),
    UmbrellaAnchor("U3", ["look at your hands", "have a look at your hands",
                           "can i see your hands", "lift them up", "check your hands"], confidence=1.0),
    UmbrellaAnchor("U4", ["take off your specs", "pull down on your eyelid",
                           "check your eyes", "have a look at your eyes"], confidence=1.0),
    UmbrellaAnchor("U5", ["feel for your heartbeat", "hand on your chest",
                           "feel for your heart beat", "placing my hand on your chest",
                           "feeling your chest wall"], confidence=1.0),
    UmbrellaAnchor("U6", ["inspection of the back", "listen to your back",
                           "no visible deformities"], confidence=1.0),
    UmbrellaAnchor("U6", ["press on your legs", "pressing on your legs"], confidence=1.0),
    UmbrellaAnchor("U6", ["end of my examination", "come to the end"], confidence=1.0),
]

STEP_ANCHORS: List[StepAnchor] = [
    StepAnchor("step_1_Greet_Patient", ["how do i address", "hi ", "hello"], "start_instruction", -0.5, 0.8),
    StepAnchor("step_2_Obtain_Consent", ["cardiovascular exam", "okay to proceed", "would that be okay", "shall we proceed"], "start_instruction", 0.0),
    StepAnchor("step_4_Explain_Exam", ["i'm here to do", "ill be doing", "i will be doing"], "start_instruction", 0.0, 0.8),
    StepAnchor("step_6_Patient_Comfort", ["are you comfortable", "comfortable?"], "start_instruction", 0.0),
    StepAnchor("step_7_Position_45_Degrees", ["45 degrees", "forty five degrees"], "start_instruction", 0.0),
    StepAnchor("step_5_Hand_Hygiene", ["clean my hands", "hand rub", "sanitise", "sanitize"], "start_instruction", 0.3),
    StepAnchor("step_8_Expose_Chest_Legs", ["expose your chest", "remove your shirt"], "start_instruction", 0.2),
    StepAnchor("step_9_General_Inspection", ["general inspection", "comfortable at rest"], "start_instruction", 0.6),
    StepAnchor("step_10_1_Inspect_Syndromic_General", ["marfan", "turner", "syndromic", "acromegaly", "trisomy"], "start_instruction", 0.0),
    StepAnchor("step_11_Inspect_Anterior_Chest", ["raise your arms", "near inspection", "scars on the chest", "pacemaker", "apex beat"], "start_instruction", 3.5),
    StepAnchor("step_12_Cover_Chest", ["cover yourself", "if you're cold", "cover up so"], "start_instruction", -1.7),
    StepAnchor("step_10_2_Inspect_Syndromic_Legs", ["scars on the legs", "no surgical scars on the legs"], "end_finding", 0.0),
    StepAnchor("step_13_Inspect_Hands", ["look at your hands", "see your hands", "lift them up"], "start_instruction", -2.4),
    StepAnchor("step_15_Palpate_Radial_Pulse", ["feel your pulse", "going to feel"], "start_instruction", 0.3),
    StepAnchor("step_15_Palpate_Radial_Pulse", ["heart rate is", "pulse rate is", "respiratory rate is"], "end_finding", 5.0),
    StepAnchor("step_17_Radio_Femoral_Delay", ["omit", "radio-femoral", "radio femoral"], "omission", 0.0),
    StepAnchor("step_14_Inspect_Forearms", ["forearms", "antecubital", "needle marks", "tendon xanthoma"], "start_instruction", 8.0),
    StepAnchor("step_18_Collapsing_Pulse", ["pain in your shoulder", "pain in shoulder", "going to lift your", "raise your arm"], "start_instruction", -0.5, 1.0),
    StepAnchor("step_18_Collapsing_Pulse", ["no collapsing pulse"], "end_finding", 0.5),
    StepAnchor("step_19_Inspect_Face", ["take off your specs", "pull down on your eyelid", "check your eyes"], "start_instruction", 2.5),
    StepAnchor("step_19_Inspect_Face", ["open your mouth", "raise your tongue", "lift up your tongue"], "start_instruction", 0.0),
    StepAnchor("step_20_1_Inspect_Neck_Corrigan", ["feel of your neck", "feel your neck", "corrigan"], "start_instruction", 0.9),
    StepAnchor("step_21_Auscultate_Carotids", ["listen to your neck", "no bruit"], "end_finding", 2.0),
    StepAnchor("step_22_Re_expose_Chest", ["expose again your chest", "expose your chest again"], "start_instruction", 0.0),
    StepAnchor("step_20_2_Inspect_JVP", ["jvp", "jugular venous", "root of the neck"], "end_finding", 5.0),
    StepAnchor("step_20_3_Abdominojugular_Reflux", ["press down on your tummy", "pressing on your tummy", "press down on your abdomen"], "start_instruction", 6.8),
    StepAnchor("step_20_3_Abdominojugular_Reflux", ["hepatojugular", "abdominojugular"], "end_finding", 13.0),
    StepAnchor("step_23_1_Palpate_Apex_Location", ["feel for your heartbeat", "hand on your chest", "placing my hand on your chest"], "start_instruction", 0.0),
    StepAnchor("step_27_1_Explain_Auscultation_Steps",
               ["listen to your heart", "listening to your heart", "listen to your chest",
                "listening to your chest", "to your heart", "listen to the heart"], "start_instruction", 0.0),
    StepAnchor("step_30_1_Turn_Left_Lateral", ["turn to your left", "turn to the left"], "start_instruction", -0.5, 1.0),
    StepAnchor("step_30_4_Auscultate_Breathing_Maneuver", ["deep breath in", "breathe in, breathe out", "hold your breath"], "start_instruction", 0.0),
    StepAnchor("step_31_1_Turn_Back_Supine", ["turn back", "back on your back"], "start_instruction", -0.5, 1.0),
    StepAnchor("step_35_1_Sit_Up_Lean_Forward", ["sit up", "sitting up"], "start_instruction", 0.0, 1.0),
    StepAnchor("step_35_1_Sit_Up_Lean_Forward", ["lean forward", "lean a bit forward"], "start_instruction", 0.0, 1.0),
    StepAnchor("step_36_Inspect_Back", ["inspection of the back", "inspect the back", "on inspection of the back"], "start_instruction", -0.5),
    StepAnchor("step_37_Auscultate_Lung_Bases", ["listen to your back", "listen to your chest", "deep breaths in and out"], "start_instruction", -2.0),
    StepAnchor("step_38_1_Inspect_Legs_Edema", ["press on your legs", "pressing on your legs"], "start_instruction", 5.6),
    StepAnchor("step_39_Thank_Patient_Cover", ["end of my examination", "thank you very much", "you can cover up"], "start_instruction", 0.0),
    StepAnchor("step_40_Final_Summary", ["temperature chart", "urine", "fundoscopy", "blood pressure",
                                         "vital signs", "parameter chart"], "start_instruction", 0.0),
]

# ==========================================
# Required keypoint CSV columns
#
# NOTE: this deliberately excludes step_* columns and Audio_Umbrella_Prediction
# -- see module docstring. Built from the 204 Front/Side/360 x Doctor/Patient
# keypoint columns plus the engineered feature columns features.py always
# produces (posture, touch, position, velocity, anatomical zones).
# ==========================================
_KEYPOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]

REQUIRED_KEYPOINT_COLUMNS: set = {"Frame"}
for _view in ("Front", "Side", "360"):
    for _role in ("Doctor", "Patient"):
        for _kpt in _KEYPOINT_NAMES:
            REQUIRED_KEYPOINT_COLUMNS.add(f"{_view}_{_role}_{_kpt}_X")
            REQUIRED_KEYPOINT_COLUMNS.add(f"{_view}_{_role}_{_kpt}_Y")

REQUIRED_KEYPOINT_COLUMNS |= {
    "Spine_Angle", "Shoulder_Tilt", "Shape", "Twist", "L_Is_Touching", "R_Is_Touching",
    "Doctor_Position", "L_Wrist_Max_Velocity", "R_Wrist_Max_Velocity",
    "Doctor_Overall_Hand_Movement", "Min_Dist_to_Chest", "Min_Dist_to_Neck",
    "Min_Dist_to_PatWrists", "L_Side_T", "L_Side_Zone", "R_Side_T", "R_Side_Zone",
    "L_Front_X", "L_Front_Zone", "R_Front_X", "R_Front_Zone", "Min_Pulse_Dist",
    "L_Combined_Zone", "R_Combined_Zone",
}


# ==========================================
# Embedding backend
# ==========================================
class EmbedBackend:
    """Wraps Gemini SDK or sentence-transformers. embed(texts) -> (N, D) float32, L2-normalised."""

    def __init__(self, backend: str = "auto", logger: Optional[logging.Logger] = None):
        self._log = logger or logging.getLogger("phase0")
        self._backend = self._resolve(backend)

    def _resolve(self, requested: str) -> str:
        if requested in ("gemini", "auto"):
            try:
                import google.genai  # noqa: F401
                if os.environ.get("GEMINI_API_KEY"):
                    self._log.info("  Embedding backend: Gemini text-embedding-004")
                    return "gemini"
                self._log.warning("  google-genai installed but GEMINI_API_KEY not set — trying local")
            except ImportError:
                if requested == "gemini":
                    raise RuntimeError("pip install google-genai")

        if requested in ("local", "auto"):
            try:
                import sentence_transformers  # noqa: F401
                self._log.info("  Embedding backend: sentence-transformers all-MiniLM-L6-v2 (local)")
                return "local"
            except ImportError:
                if requested == "local":
                    raise RuntimeError("pip install sentence-transformers")

        raise RuntimeError(
            "No embedding backend available.\n"
            "  pip install google-genai        (+ set GEMINI_API_KEY via Colab Secrets)\n"
            "  pip install sentence-transformers"
        )

    @property
    def name(self) -> str:
        return self._backend

    @property
    def backend_id(self) -> str:
        return {
            "gemini": "gemini:text-embedding-004",
            "local": "sentence-transformers:all-MiniLM-L6-v2",
        }[self._backend]

    @property
    def dim(self) -> int:
        return {"gemini": 768, "local": 384}[self._backend]

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._embed_gemini(texts) if self._backend == "gemini" else self._embed_local(texts)

    def _embed_gemini(self, texts: List[str]) -> np.ndarray:
        from google import genai
        client = genai.Client()
        resp = client.models.embed_content(model="text-embedding-004", contents=texts)
        vecs = np.array([item.values for item in resp.embeddings], dtype=np.float32)
        return vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, "_model"):
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vecs.astype(np.float32)


def build_anchor_index() -> List[dict]:
    records: List[dict] = []
    for ua in UMBRELLA_ANCHORS:
        for phrase in ua.phrases:
            records.append({
                "type": "umbrella", "umbrella": ua.umbrella, "step": None,
                "phrase": phrase, "role": ua.role,
                "lead_lag_sec": 0.0, "confidence": ua.confidence, "required": ua.required,
            })
    for sa in STEP_ANCHORS:
        umbrella = next((uid for uid, steps in UMBRELLA_STEPS.items() if sa.step in steps), None)
        for phrase in sa.phrases:
            records.append({
                "type": "step", "umbrella": umbrella, "step": sa.step,
                "phrase": phrase, "role": sa.role,
                "lead_lag_sec": sa.lead_lag_sec, "confidence": sa.confidence, "required": False,
            })
    return records


def embed_anchors(
    backend: EmbedBackend,
    logger: logging.Logger,
    cache_path: Optional[str] = None,
    index_cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, List[dict]]:
    index = build_anchor_index()
    phrases = [r["phrase"] for r in index]

    logger.info(f"── Anchor embedding  [{len(phrases)} phrases] ──────────────")

    if cache_path and index_cache_path and \
       os.path.exists(cache_path) and os.path.exists(index_cache_path):
        with open(index_cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if [r["phrase"] for r in cached] == phrases:
            matrix = np.load(cache_path)
            logger.info(f"  ✓  Cache hit: {os.path.basename(cache_path)}  shape={matrix.shape}")
            return matrix, cached
        logger.info("  Phrase list changed — re-embedding")

    logger.info(f"  Backend: {backend.name}  dim={backend.dim}")
    batches = [phrases[i:i + 64] for i in range(0, len(phrases), 64)]
    rows: List[np.ndarray] = []
    for i, batch in enumerate(batches):
        logger.info(f"    batch {i + 1}/{len(batches)}  ({len(batch)} phrases)")
        rows.append(backend.embed(batch))
    matrix = np.vstack(rows).astype(np.float32)

    if cache_path and index_cache_path:
        np.save(cache_path, matrix)
        with open(index_cache_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        logger.info(f"  ✓  Saved: {os.path.basename(cache_path)}  shape={matrix.shape}")

    logger.info(f"  ✓  Done  shape={matrix.shape}")
    return matrix, index


# ==========================================
# Validation
# ==========================================
def validate_taxonomy(logger: logging.Logger) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    logger.info("── Taxonomy validation ─────────────────────────────────")

    all_steps: Dict[str, str] = {}
    for uid in UMBRELLA_ORDER:
        if uid not in UMBRELLA_STEPS:
            errors.append(f"{uid} in UMBRELLA_ORDER but missing from UMBRELLA_STEPS")
            continue
        for s in UMBRELLA_STEPS[uid]:
            if s in all_steps:
                errors.append(f"Duplicate step '{s}' in {uid} (first: {all_steps[s]})")
            else:
                all_steps[s] = uid

    for group in CO_LOCATED_GROUPS:
        for s in group:
            if s not in all_steps:
                errors.append(f"CO_LOCATED_GROUPS: unknown step '{s}'")

    for parent, children in SPLITTABLE_STEPS.items():
        for s in children:
            if s not in all_steps:
                errors.append(f"SPLITTABLE_STEPS '{parent}': unknown step '{s}'")

    unknown_anchor_steps = {sa.step for sa in STEP_ANCHORS} - set(all_steps)
    if unknown_anchor_steps:
        errors.append(f"STEP_ANCHORS reference unknown steps: {sorted(unknown_anchor_steps)}")

    unknown_umbrellas = {ua.umbrella for ua in UMBRELLA_ANCHORS} - set(UMBRELLA_ORDER)
    if unknown_umbrellas:
        errors.append(f"UMBRELLA_ANCHORS reference unknown umbrellas: {sorted(unknown_umbrellas)}")

    total_phrases = (sum(len(ua.phrases) for ua in UMBRELLA_ANCHORS) +
                     sum(len(sa.phrases) for sa in STEP_ANCHORS))
    logger.info(f"  Umbrellas: {len(UMBRELLA_ORDER)}   Steps: {len(all_steps)}   "
                f"Anchor phrases: {total_phrases}")
    for uid in UMBRELLA_ORDER:
        steps = UMBRELLA_STEPS.get(uid, [])
        n_u_anch = sum(1 for ua in UMBRELLA_ANCHORS if ua.umbrella == uid)
        n_s_anch = sum(1 for sa in STEP_ANCHORS if sa.step in steps)
        logger.info(f"    {uid}  {UMBRELLA_DESCRIPTIONS[uid]:<30}  {len(steps):>2} steps  "
                    f"{n_u_anch} umbrella anchors  {n_s_anch} step anchors")

    if errors:
        for e in errors:
            logger.error(f"  FAIL  {e}")
        return False, errors
    logger.info("  ✓ Taxonomy OK")
    return True, []


def validate_files(
    video_paths: Dict[str, str],
    csv_path: str,
    logger: logging.Logger,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    logger.info("── File existence check ────────────────────────────────")

    def check(label: str, path: str) -> None:
        if not os.path.exists(path):
            errors.append(f"Missing {label}: {path}")
            logger.error(f"  MISS  {label}: {os.path.basename(path)}")
        elif os.path.getsize(path) == 0:
            errors.append(f"Empty {label}: {path}")
            logger.error(f"  EMPTY {label}: {os.path.basename(path)}")
        else:
            size_mb = os.path.getsize(path) / 1e6
            logger.info(f"  ✓  {label}: {os.path.basename(path)}  ({size_mb:.1f} MB)")

    for cam, path in video_paths.items():
        check(f"video [{cam}]", path)

    check("wide keypoints CSV", csv_path)

    if errors:
        return False, errors
    logger.info("  ✓ All files present")
    return True, []


def validate_keypoint_columns(
    csv_path: str,
    logger: logging.Logger,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    logger.info("── Keypoint CSV schema ─────────────────────────────────")

    if not os.path.exists(csv_path):
        return False, [f"Missing Wide CSV: {csv_path}"]

    with open(csv_path, newline="", encoding="utf-8") as f:
        header = set(next(csv.reader(f), []))

    missing = REQUIRED_KEYPOINT_COLUMNS - header
    if missing:
        missing_list = sorted(list(missing))
        errors.append(f"Wide CSV missing {len(missing)} columns (e.g., {missing_list[:5]})")
        logger.error(f"  FAIL  Wide CSV missing {len(missing)} columns. Examples: {missing_list[:5]}...")
    else:
        logger.info(f"  ✓  Wide CSV schema valid ({len(header)} columns verified)")

    if errors:
        return False, errors
    logger.info("  ✓ CSV schema check passed")
    return True, []


def validate_dependencies(logger: logging.Logger) -> Tuple[bool, dict]:
    errors: List[str] = []
    dep: dict = {}
    logger.info("── System dependencies ─────────────────────────────────")

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            errors.append(f"{tool} not on PATH")
            logger.error(f"  MISS  {tool}")
        else:
            logger.info(f"  ✓  {tool}")

    dep["numpy"] = np.__version__
    logger.info(f"  ✓  numpy: {dep['numpy']}")

    has_embed = False
    for pkg, label in [("google.genai", "google-genai"), ("sentence_transformers", "sentence-transformers")]:
        try:
            mod = __import__(pkg.split(".")[0])
            dep[pkg] = getattr(mod, "__version__", "installed")
            has_embed = True
            logger.info(f"  ✓  {label}: {dep[pkg]}")
        except ImportError:
            logger.info(f"  –  {label} not installed")

    if not has_embed:
        errors.append("No embedding backend: pip install sentence-transformers or google-genai")
        logger.error("  MISS  embedding backend")

    for pkg, pip in [("cv2", "opencv-python-headless"), ("PIL", "Pillow")]:
        try:
            mod = __import__(pkg)
            dep[pkg] = getattr(mod, "__version__", "installed")
            logger.info(f"  ✓  {pkg}: {dep[pkg]}")
        except ImportError:
            errors.append(f"{pip} not installed")
            logger.error(f"  MISS  {pkg}")

    if errors:
        return False, {}
    logger.info("  ✓ All dependencies OK")
    return True, dep


def validate_video_metadata(
    video_paths: Dict[str, str],
    logger: logging.Logger,
) -> dict:
    logger.info("── Video metadata ──────────────────────────────────────")
    metadata: dict = {}
    for cam, path in video_paths.items():
        if not os.path.exists(path):
            continue
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,duration,r_frame_rate",
               "-of", "json", path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        try:
            stream = json.loads(result.stdout).get("streams", [{}])[0]
            dur = float(stream.get("duration", 0))
            w, h = stream.get("width", "?"), stream.get("height", "?")
            num, den = stream.get("r_frame_rate", "0/1").split("/")
            fps = round(int(num) / int(den), 2) if int(den) else 0
            metadata[cam] = {"duration_sec": round(dur, 2), "resolution": f"{w}x{h}", "fps": fps}
            logger.info(f"  ✓  [{cam}]  {dur:.1f}s  {w}x{h}  {fps}fps")
        except Exception as exc:
            logger.warning(f"  WARN [{cam}] ffprobe: {exc}")

    durs = [v["duration_sec"] for v in metadata.values()]
    if len(durs) > 1 and max(durs) - min(durs) > 5:
        logger.warning(f"  WARN  Camera duration spread {max(durs) - min(durs):.1f}s > 5s")
    return metadata


# ==========================================
# Config assembly + orchestrator
# ==========================================
def build_phases_config(
    take_id: int,
    proc_root: str,
    video_paths: Dict[str, str],
    csv_path: str,
    video_metadata: dict,
    dep_info: dict,
    embed_matrix: np.ndarray,
    anchor_index: List[dict],
    backend: EmbedBackend,
    anchor_embeddings_path: str,
    anchor_index_path: str,
) -> dict:
    splittable_all = {s for children in SPLITTABLE_STEPS.values() for s in children}

    umbrellas = []
    for uid in UMBRELLA_ORDER:
        steps = UMBRELLA_STEPS[uid]
        substeps = []
        for sname in steps:
            s_anchors = [
                {"phrases": sa.phrases, "role": sa.role,
                 "lead_lag_sec": sa.lead_lag_sec, "confidence": sa.confidence}
                for sa in STEP_ANCHORS if sa.step == sname
            ]
            co_group = next((g for g in CO_LOCATED_GROUPS if sname in g), None)
            substeps.append({
                "step_id": sname, "umbrella": uid, "anchors": s_anchors,
                "co_located_group": co_group, "is_splittable": sname in splittable_all,
            })
        umbrellas.append({
            "id": uid, "name": UMBRELLA_DESCRIPTIONS[uid], "step_count": len(steps),
            "anchors": [
                {"phrases": ua.phrases, "role": ua.role,
                 "confidence": ua.confidence, "required": ua.required}
                for ua in UMBRELLA_ANCHORS if ua.umbrella == uid
            ],
            "steps": substeps,
        })

    return {
        "schema_version": "2.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "dep_versions": dep_info,
            "proc_root": proc_root,
            "reference_take_id": str(take_id),
            "embed_backend": backend.backend_id,
            "embed_dim": int(embed_matrix.shape[1]) if embed_matrix.ndim > 1 else 0,
            "anchor_phrase_count": int(embed_matrix.shape[0]),
            "anchor_embeddings_file": anchor_embeddings_path,
            "anchor_index_file": anchor_index_path,
        },
        "reference_files": {"videos": video_paths, "wide_csv": csv_path, "video_metadata": video_metadata},
        "pipeline_config": PIPELINE_CONFIG,
        "co_located_groups": CO_LOCATED_GROUPS,
        "splittable_steps": [s for children in SPLITTABLE_STEPS.values() for s in children],
        "weak_sequential_fallback": [list(p) for p in WEAK_SEQUENTIAL_FALLBACK],
        "umbrellas": umbrellas,
    }


def setup_logging(log_path: str) -> logging.Logger:
    log = logging.getLogger("phase0")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    fmt = logging.Formatter("%(levelname)-8s  %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    log.addHandler(ch)
    log.addHandler(fh)
    return log


def run_phase0(take_id: int, embed_backend: str = "local", skip_embed: bool = False) -> int:
    """Validate a reference take's inputs and write phases_config.json +
    anchor_embeddings.npy + anchor_index.json to its processed folder.

    Returns 0 on success, 1 if any check failed (nothing is written on failure).
    """
    p = config.get_take_paths(take_id)
    logger = setup_logging(p["phase0_validation_log"])
    logger.info(f"CVS Pipeline — Phase 0   [{datetime.now(timezone.utc).isoformat()}]")
    logger.info(f"Reference take  : Take_{take_id}")
    logger.info(f"Processed root  : {p['processed_folder']}")
    logger.info(f"Embed backend   : {embed_backend}")
    logger.info("")

    video_paths: Dict[str, str] = {"front": p["vid_front"], "side": p["vid_side"], "360": p["vid_360"]}
    wide_csv_path: str = p["wide_keypoints_csv"]

    all_errors: List[str] = []

    taxonomy_ok, e = validate_taxonomy(logger)
    all_errors.extend(e)

    files_ok, e = validate_files(video_paths, wide_csv_path, logger)
    all_errors.extend(e)

    columns_ok, e = validate_keypoint_columns(wide_csv_path, logger)
    all_errors.extend(e)

    deps_ok, dep_info = validate_dependencies(logger)
    if not deps_ok:
        all_errors.append("System dependencies missing — see log")

    video_metadata = {}
    if shutil.which("ffprobe") and files_ok:
        video_metadata = validate_video_metadata(video_paths, logger)

    embed_matrix: Optional[np.ndarray] = None
    anchor_index: List[dict] = []
    backend: Optional[EmbedBackend] = None
    embed_shape = None

    if not all_errors and not skip_embed:
        try:
            backend = EmbedBackend(embed_backend, logger)
            embed_matrix, anchor_index = embed_anchors(
                backend, logger,
                cache_path=p["anchor_embeddings"],
                index_cache_path=p["anchor_index"],
            )
            embed_shape = tuple(embed_matrix.shape)
        except Exception as exc:
            all_errors.append(f"Embedding failed: {exc}")
            logger.error(f"  FAIL  {exc}")
    elif skip_embed:
        logger.info("── Embedding skipped (skip_embed=True) ────────────────")
        anchor_index = build_anchor_index()
        embed_matrix = np.zeros((len(anchor_index), 1), dtype=np.float32)

        class _StubBackend:
            backend_id = "skipped"
            dim = 0
        backend = _StubBackend()

    if not all_errors and embed_matrix is not None and backend is not None:
        phases_cfg = build_phases_config(
            take_id=take_id,
            proc_root=p["processed_folder"],
            video_paths=video_paths,
            csv_path=wide_csv_path,
            video_metadata=video_metadata,
            dep_info=dep_info,
            embed_matrix=embed_matrix,
            anchor_index=anchor_index,
            backend=backend,
            anchor_embeddings_path=p["anchor_embeddings"],
            anchor_index_path=p["anchor_index"],
        )
        with open(p["phases_config"], "w", encoding="utf-8") as f:
            json.dump(phases_cfg, f, indent=2)

        if not skip_embed:
            np.save(p["anchor_embeddings"], embed_matrix)
            with open(p["anchor_index"], "w", encoding="utf-8") as f:
                json.dump(anchor_index, f, indent=2)

    if not all_errors:
        logger.info(f"\n✅ Phase 0 complete.  Outputs written to:\n   {p['processed_folder']}")
    else:
        logger.info(f"\n❌ Phase 0 finished with {len(all_errors)} error(s).")
        for e in all_errors:
            logger.error(f"    • {e}")

    return 0 if not all_errors else 1
