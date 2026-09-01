"""
Stage 01 (Track A): constraint-aware Viterbi step decoder.

Fresh module -- not ported from any notebook. Implements Constraint-Aware
Decoding (CAD, arXiv:2605.10149, "Improving Temporal Action Segmentation via
Constraint-Aware Decoding") adapted to our *alignment* setting (the 56-step
checklist order is known at inference time, so this isn't open-set
segmentation) -- see Obsidian/segmentation_decoder_build.md and
Obsidian/Segmentation_Grading_Strategy_Notes.md sections 2/8 for the design
history and locked-in decisions this file assumes:

  - Duration bounds are a tunable hyperparameter (estimate_duration_bounds()),
    not literature constants -- we only have 1-2 annotated takes to fit from.
  - Transition constraints are SOFT and order-distance-based
    (transition_penalty()), not a hard valid/invalid transition matrix like
    the paper's Conf(A->B) -- a hard set would make legitimate late step
    resumption structurally inadmissible.
  - Checklist order = flattening phase0.UMBRELLA_STEPS in phase0.UMBRELLA_ORDER
    order (STEP_ORDER below), which already reflects real empirical step
    ordering from the reference take, not raw checklist numbering.
  - CO_LOCATED_GROUPS (steps that are genuinely simultaneous, e.g. the
    greeting cluster 1/2/3/4/6/7, confirmed identical-timestamped in
    annotations_master.csv) are merged into single meta-states for decoding
    and expanded back to per-step segments afterward -- a flat 56-state
    Viterbi can't represent "several steps are true at once" since it
    assigns one label per frame. Because group members are merged BEFORE
    decoding, they share the exact same decoded window by construction --
    there is no separate "enforce concurrency" post-processing step needed.
  - Our valid start/end sets (CAD section 3.1) degenerate to the singleton
    first/last meta-state in STEP_ORDER: CAD normally mines S/E from many
    training sequences that can start/end on different actions; we have one
    fixed, known checklist order, so there's exactly one valid start and one
    valid end.

EMISSION SIGNAL has two additive components (see build_emission_matrix()):
  1. Transcript-anchor cosine-similarity bumps, for the 34/56 steps with a
     phase0.STEP_ANCHORS entry.
  2. Pose anatomical-zone matches (build_zone_emission()), for steps in
     STEP_ZONE_HINTS -- mostly the ~22 unanchored, acoustically-silent
     fine-grained U5 palpation/auscultation steps (25_1-25_4 thrill sites,
     26_1/26_2 P2/A2, and the auscultation-site sequence 27_2-35_2) that
     transcript alone cannot distinguish, confirmed against
     annotations_master.csv (all their rows are silent, no distinguishing
     narration). Uses features.py's L/R_Combined_Zone (coarse
     Left/Center/Right x Neck/Upper_Chest/Lower_Chest/Stomach/Pelvis
     doctor-palm-vs-patient-torso projection) -- NOT true anatomical
     landmarks (apex/pulmonary/aortic auscultation points), so
     STEP_ZONE_HINTS is a coarse, hand-authored approximation (e.g. apex ~
     Left_Lower_Chest, pulmonary/P2 ~ Left_Upper_Chest, aortic/A2 ~
     Right_Upper_Chest) that needs real-data confirmation on Colab, not a
     validated clinical mapping. Several hint steps intentionally share a
     zone (e.g. apex location/character/heave/thrill are all
     Left_Lower_Chest) -- zone signal only separates *sites*, duration +
     transition ordering still does all within-site sequencing. Weighted
     separately (DEFAULT_ZONE_WEIGHT) so it can be dialed to zero without
     touching the transcript-anchor path if it doesn't hold up on real data.

STATUS: implemented and locally validated against synthetic data (see
Obsidian/segmentation_decoder_build.md test log) covering the three real
challenge patterns actually present in Take 3's own ground truth: concurrent
steps (the greeting cluster), out-of-order resumption (step_14 is
genuinely "done out of checklist order" per annotations_master.csv), and
fragmented/revisited steps (step_28_2 has two disjoint windows). Parsing
validated against the real Take 3 transcript and annotations_master.csv.
NOT YET RUN against real embeddings or real pose zone data on Colab --
STEP_ZONE_HINTS and all hyperparameters below are unvalidated guesses,
deliberately wide/tunable, not literature- or corpus-fit.

KNOWN GAPS / assumptions not yet verified on real data:
  - STEP_ZONE_HINTS's zone->step mapping (anatomical judgment call, not
    measured).
  - Whether transcript+zone emission together are actually sufficient to
    resolve 25_1-25_4 (~1-2s each) at 24fps precision, vs. just being a
    marginal improvement over flat -- only real Colab data will show this.
  - Duration bounds fit-then-eval circularity when duration_reference_take_id
    is left as the take being decoded (only Take 3 has full annotations
    right now -- Take 2's manual_annotations_take2.csv lives on Drive only).
  - All hyperparameters (DEFAULT_SIGMA_SEC, DEFAULT_FORWARD_RATE,
    DEFAULT_BACKWARD_RATE, DEFAULT_DURATION_SHRINK/GROW, DEFAULT_ZONE_WEIGHT)
    are wide starting guesses for manual sweeping, not literature values.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from . import config, phase0

# ==========================================
# Tunable hyperparameters (start wide -- see module docstring)
# ==========================================
# Anchor bump width. Measured, not a guess. Originally
# PIPELINE_CONFIG["boundary_slop_sec"]/3 = 10.0s, but boundary_slop is an
# evaluation tolerance and was never meant as an emission width; 10s is far
# wider than the ~1-2s steps it must separate, and is clearly worst on BOTH
# takes. Briefly set to 2.0 on 2026-09-01 from a Take-3-only sweep -- that
# comparison was CONFOUNDED (the sigma=1.0 arm had the zone signal on).
# Re-swept 2026-09-02 with zone off on both takes, mean over Take 2 + Take 3:
#     sigma 1.0 -> IoU 0.200  F1@50 10.4  frameAcc 41.2   <- best
#     sigma 2.0 -> IoU 0.180  F1@50  9.6  frameAcc 39.6
#     sigma 3.0 -> IoU 0.184  F1@50  9.8  frameAcc 41.7
#     sigma 4.0 -> IoU 0.165  F1@50  9.5  frameAcc 39.2
# 1.0 and 2.0 are a wash on Take 3 alone, but 1.0 is much better on Take 2
# (IoU 0.182 vs 0.145), so 1.0 wins overall. Still only n=2 takes.
DEFAULT_SIGMA_SEC = 1.0
DEFAULT_FORWARD_RATE = 0.15    # soft transition penalty, per order-step skipped ahead
DEFAULT_BACKWARD_RATE = 0.6    # soft transition penalty, per order-step resumed backward
DEFAULT_DURATION_SHRINK = 0.4  # d_min = observed_duration * shrink
DEFAULT_DURATION_GROW = 2.5    # d_max = observed_duration * grow
DEFAULT_DURATION_FALLBACK = (0.001, 0.25)  # used for steps missing from the reference annotations
# Relative weight of zone-match emission vs. transcript-anchor emission.
# DEFAULTS TO 0.0 = the zone signal is OFF, because it measurably HURTS.
# Ablated on real Take 3 (2026-09-01): at every sigma tested, zone_weight 0.0
# beat 1.0 beat 3.0 on F1 and frame accuracy -- monotonic, not noise. At
# sigma=2.0: F1@50 13.0 (off) vs 8.5 (1.0) vs 6.6 (3.0). STEP_ZONE_HINTS is
# hand-authored anatomical guesswork and does not survive contact with real
# data. The code path is kept, not deleted, so the mapping can be REBUILT from
# measured L_Combined_Zone/R_Combined_Zone values and re-ablated -- but do not
# simply tune this value up; the mapping itself is what's wrong.
# NB: IoU alone mildly *preferred* the zone signal. Judge it on F1/edit.
DEFAULT_ZONE_WEIGHT = 0.0

NEG_INF = -1e18


# ==========================================
# Canonical step order + co-located meta-states
# ==========================================
def flatten_step_order() -> List[str]:
    order: List[str] = []
    for umbrella in phase0.UMBRELLA_ORDER:
        order.extend(phase0.UMBRELLA_STEPS[umbrella])
    return order


STEP_ORDER: List[str] = flatten_step_order()


@dataclass
class MetaState:
    id: int
    members: List[str]  # >1 step id only for a merged CO_LOCATED_GROUPS group

    @property
    def label(self) -> str:
        return "+".join(self.members)


def build_meta_states(
    step_order: Sequence[str] = STEP_ORDER,
    co_located_groups: Sequence[Sequence[str]] = phase0.CO_LOCATED_GROUPS,
) -> List[MetaState]:
    """Collapse each CO_LOCATED_GROUPS group into one meta-state, in the
    order its first member appears in step_order. Everything else stays a
    singleton meta-state. Decoding happens at meta-state granularity;
    path_to_segments() expands merged states back to per-step segments."""
    step_to_group: Dict[str, int] = {}
    for gi, group in enumerate(co_located_groups):
        for step in group:
            step_to_group[step] = gi

    meta_states: List[MetaState] = []
    seen_groups: set = set()
    for step in step_order:
        if step in step_to_group:
            gi = step_to_group[step]
            if gi in seen_groups:
                continue
            seen_groups.add(gi)
            members = [s for s in co_located_groups[gi] if s in step_order]
            meta_states.append(MetaState(id=len(meta_states), members=members))
        else:
            meta_states.append(MetaState(id=len(meta_states), members=[step]))
    return meta_states


def step_to_meta_index(meta_states: List[MetaState]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for ms in meta_states:
        for step in ms.members:
            mapping[step] = ms.id
    return mapping


def meta_label_for_step(step: str, meta_states: List[MetaState]) -> str:
    for ms in meta_states:
        if step in ms.members:
            return ms.label
    return step


# ==========================================
# Transcript parsing
# ==========================================
_UTTERANCE_RE = re.compile(r"^\*\*(\d{1,3}):(\d{2})\s+([^:*]+?):\*\*\s*(.*)$")


@dataclass
class Utterance:
    start_sec: float
    speaker: str
    text: str


def parse_transcript(path: str) -> List[Utterance]:
    """Parses the '**MM:SS Speaker:** text' format confirmed against
    Take3_Transcript_Full.txt. '**[MM:SS - MM:SS silence]**' gap markers and
    any non-matching line (e.g. the leading description line) are skipped --
    they carry no content to embed."""
    utterances: List[Utterance] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _UTTERANCE_RE.match(line.strip())
            if not m:
                continue
            mm, ss, speaker, text = m.groups()
            text = text.strip()
            if text:
                utterances.append(Utterance(
                    start_sec=float(int(mm) * 60 + int(ss)),
                    speaker=speaker.strip(),
                    text=text,
                ))
    return utterances


# ==========================================
# Anchor matching (embedding cosine similarity)
# ==========================================
@dataclass
class AnchorMatch:
    step: str
    role: str
    time_sec: float
    strength: float


def load_step_anchors(anchor_embeddings_path: str, anchor_index_path: str) -> Tuple[np.ndarray, List[dict]]:
    """Loads phase0's cached anchor_embeddings.npy/anchor_index.json and
    filters both to type=='step' rows, keeping them aligned (the cache
    interleaves umbrella and step rows in build_anchor_index()'s order)."""
    vectors = np.load(anchor_embeddings_path)
    with open(anchor_index_path, encoding="utf-8") as f:
        index = json.load(f)
    mask = np.array([r["type"] == "step" for r in index])
    step_vectors = vectors[mask]
    step_index = [r for r, m in zip(index, mask) if m]
    return step_vectors, step_index


def match_anchors_to_transcript(
    anchor_vectors: np.ndarray,
    anchor_index: List[dict],
    utterances: List[Utterance],
    embed_backend: phase0.EmbedBackend,
    sim_floor: float = phase0.PIPELINE_CONFIG["anchor_sim_low"],
) -> List[AnchorMatch]:
    """For each step-anchor phrase, finds its best-matching transcript
    utterance by cosine similarity and, if above sim_floor, emits an
    AnchorMatch at utterance_start + lead_lag_sec. No speaker filtering --
    patient one-word replies score far below sim_floor against clinical
    phrases naturally, and no per-take 'who is the doctor' field exists in
    config.py to filter on generically."""
    if not utterances:
        return []
    utter_vectors = embed_backend.embed([u.text for u in utterances])
    sims = anchor_vectors @ utter_vectors.T  # (A, U), both L2-normalised -> cosine sim

    matches: List[AnchorMatch] = []
    for a_idx, rec in enumerate(anchor_index):
        u_idx = int(np.argmax(sims[a_idx]))
        sim = float(sims[a_idx, u_idx])
        if sim < sim_floor:
            continue
        adjusted_time = utterances[u_idx].start_sec + rec["lead_lag_sec"]
        matches.append(AnchorMatch(
            step=rec["step"], role=rec["role"],
            time_sec=max(0.0, adjusted_time),
            strength=sim * rec["confidence"],
        ))
    return matches


# ==========================================
# Pose anatomical-zone hints (see module docstring -- unvalidated on real data)
# ==========================================
STEP_ZONE_HINTS: Dict[str, List[str]] = {
    "step_23_1_Palpate_Apex_Location": ["Left_Lower_Chest"],
    "step_23_2_Palpate_Apex_Character": ["Left_Lower_Chest"],
    "step_24_Palpate_Heave": ["Left_Lower_Chest", "Center_Lower_Chest"],
    "step_25_1_Thrill_Apex": ["Left_Lower_Chest"],
    "step_25_2_Thrill_Left_Sternal": ["Center_Lower_Chest", "Left_Lower_Chest"],
    "step_25_3_Thrill_Pulmonary": ["Left_Upper_Chest"],
    "step_25_4_Thrill_Aortic": ["Right_Upper_Chest"],
    "step_26_1_Palpate_P2": ["Left_Upper_Chest"],
    "step_26_2_Palpate_A2": ["Right_Upper_Chest"],
    "step_27_2_Auscultate_Mitral_Diaphragm": ["Left_Lower_Chest"],
    "step_29_2_Auscultate_Mitral_Bell": ["Left_Lower_Chest"],
    "step_30_2_Palpate_Apex_Lateral": ["Left_Lower_Chest"],
    "step_30_3_Auscultate_Mitral_Bell_Lateral": ["Left_Lower_Chest"],
    "step_31_2_Auscultate_Tricuspid": ["Center_Lower_Chest", "Left_Lower_Chest"],
    "step_32_Auscultate_Pulmonary": ["Left_Upper_Chest"],
    "step_33_Auscultate_Aortic": ["Right_Upper_Chest"],
    "step_34_Auscultate_Carotid_Bruits": ["Neck"],
    "step_35_2_Auscultate_Aortic_Regurgitation": ["Center_Lower_Chest", "Left_Lower_Chest"],
}


def build_zone_emission(
    meta_states: List[MetaState],
    wide_keypoints_df: pd.DataFrame,
    zone_hints: Dict[str, List[str]] = STEP_ZONE_HINTS,
) -> np.ndarray:
    """(T, M) raw emission contribution from doctor-palm anatomical zone
    matches. Flat +1 bump on every frame where either hand's Combined_Zone
    (features.py) is in the step's hint list -- deliberately not a Gaussian
    (unlike the transcript-anchor bumps): there's no single timestamp to
    center on, just a binary "hand is plausibly at this site" signal, so
    duration bounds/self-transition already do the temporal shaping. Only
    steps present in zone_hints get anything; every other meta-state's
    column stays untouched by this function."""
    step_meta = step_to_meta_index(meta_states)
    T = len(wide_keypoints_df)
    M = len(meta_states)
    raw = np.zeros((T, M), dtype=np.float64)

    l_zone = wide_keypoints_df["L_Combined_Zone"].values
    r_zone = wide_keypoints_df["R_Combined_Zone"].values

    for step, zones in zone_hints.items():
        m_idx = step_meta.get(step)
        if m_idx is None:
            continue
        match = np.isin(l_zone, zones) | np.isin(r_zone, zones)
        raw[match, m_idx] += 1.0

    return raw


# ==========================================
# Emission matrix
# ==========================================
def build_emission_matrix(
    meta_states: List[MetaState],
    anchor_matches: List[AnchorMatch],
    total_frames: int,
    fps: float = phase0.FPS,
    sigma_sec: float = DEFAULT_SIGMA_SEC,
) -> np.ndarray:
    """(T, M) raw pre-softmax transcript-anchor emission scores. Each
    matched anchor adds a Gaussian bump (width sigma_sec) at its adjusted
    timestamp to its step's meta-state column; multiple anchors for the
    same step (e.g. a start_instruction + an end_finding) sum, reinforcing
    the interior between them. Meta-states with zero matches (the ~22/56
    steps with no STEP_ANCHORS entry at all) stay all-zero here -- see
    build_zone_emission() for the complementary pose-based signal, and the
    module docstring for why flat is the correct fallback where neither
    applies (duration bounds + transition penalty position them between
    anchored neighbors)."""
    step_meta = step_to_meta_index(meta_states)
    M = len(meta_states)
    raw = np.zeros((total_frames, M), dtype=np.float64)
    frames = np.arange(total_frames, dtype=np.float64)
    sigma_frames = max(sigma_sec * fps, 1.0)

    for match in anchor_matches:
        m_idx = step_meta.get(match.step)
        if m_idx is None:
            continue
        center_frame = match.time_sec * fps
        bump = match.strength * np.exp(-0.5 * ((frames - center_frame) / sigma_frames) ** 2)
        raw[:, m_idx] += bump

    return raw


def emission_log_probs(raw_emission: np.ndarray) -> np.ndarray:
    """Row-wise (per-frame) softmax -> log-probabilities. An all-zero row
    (no anchor/zone evidence for any meta-state at that frame) softmaxes to
    uniform, which is the intended flat prior."""
    shifted = raw_emission - raw_emission.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return np.log(probs + 1e-12)


# ==========================================
# Duration bounds
# ==========================================
def read_annotations(annotations_csv_path: str) -> "pd.DataFrame":
    """Reads a ground-truth annotations CSV and guarantees a `step_names`
    column of full step ids (e.g. `step_25_1_Thrill_Apex`).

    Two schemas for the same data exist on Drive and both are in active use,
    so this normalises rather than picking a winner:
      - `step_names` holding full step ids  (`manual_annotations_take{N}.csv`)
      - `step_id` + `name` holding the parts (`annotations_master.csv`, which
        is what config's `annotations_csv` points at, and what
        `reference_library.py` reads)
    The reconstruction rule is copied from `reference_library.py`'s
    `f"step_{row['step_id']}_{row['name']}"` so both modules agree; verified
    on Take 3 as an exact 57/57 match with identical timings between the two
    files. Without this, decoding off `annotations_master.csv` died with
    `KeyError: 'step_names'`."""
    df = pd.read_csv(annotations_csv_path)
    if "step_names" not in df.columns:
        if not ("step_id" in df.columns and "name" in df.columns):
            raise KeyError(
                f"{annotations_csv_path} has neither a 'step_names' column nor the "
                f"'step_id'+'name' pair needed to build one; columns are {list(df.columns)}"
            )
        df = df.copy()
        df["step_names"] = [f"step_{r['step_id']}_{r['name']}" for _, r in df.iterrows()]
    else:
        df = df.copy()

    # A blank start/end means the step was NOT PERFORMED. That is a real,
    # deliberate case: Take 2 is a flawed performance and has 4 such rows
    # (step_10_2, plus the whole 27_1/28_1/29_1 explain-triad, annotated "no
    # explaination done"). Coerce to NaN so timing consumers can drop them via
    # performed_rows(); before this, those blanks raised "cannot convert float
    # NaN to integer" from deep inside the metrics. The rows are KEPT here
    # rather than dropped, because their notes are grading signal.
    for col in ("start_time", "end_time"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def performed_rows(df: "pd.DataFrame") -> "pd.DataFrame":
    """Rows for steps that actually happened (real start AND end times).

    A step that was not performed contributes no ground-truth interval, which
    is the semantically right thing: a decoder that predicts a span for it is
    correctly charged a false positive, and one that predicts nothing is
    correctly unpenalised. Every timing/metric consumer filters through this;
    grading-oriented callers should use read_annotations() directly so they
    still see the not-performed rows and their notes."""
    return df[df["start_time"].notna() & df["end_time"].notna()]


def estimate_duration_bounds(
    annotations_csv_path: str,
    meta_states: List[MetaState],
    total_duration_sec: float,
    shrink: float = DEFAULT_DURATION_SHRINK,
    grow: float = DEFAULT_DURATION_GROW,
    default_frac: Tuple[float, float] = DEFAULT_DURATION_FALLBACK,
) -> List[Tuple[float, float]]:
    """Normalized [d_min, d_max] per meta-state, seeded from one annotated
    take's real durations and widened by shrink/grow. Tunable, not fit to a
    corpus -- see module docstring re: only 1-2 annotated takes existing.
    Steps absent from the CSV fall back to default_frac."""
    df = performed_rows(read_annotations(annotations_csv_path))
    step_durations: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        step = str(row["step_names"]).strip()
        dur = max(float(row["end_time"]) - float(row["start_time"]), 1e-6)
        step_durations.setdefault(step, []).append(dur)

    bounds: List[Tuple[float, float]] = []
    for ms in meta_states:
        durs: List[float] = []
        for step in ms.members:
            durs.extend(step_durations.get(step, []))
        if durs:
            frac_min = min(durs) / total_duration_sec
            frac_max = max(durs) / total_duration_sec
            d_min = max(frac_min * shrink, 1e-4)
            d_max = min(frac_max * grow, 1.0)
        else:
            d_min, d_max = default_frac
        bounds.append((d_min, d_max))
    return bounds


# ==========================================
# Constraint-aware Viterbi decoding
# ==========================================
def transition_penalty(order_dist: int, forward_rate: float, backward_rate: float) -> float:
    """Soft, order-distance-based transition penalty -- NOT a hard
    valid/invalid transition matrix (locked decision, build brief). Forward
    jumps (skipping ahead in the checklist) are cheap and grow linearly;
    backward resumptions are allowed but cost more per step of distance,
    since a trainee resuming a recently-skipped step is normal (e.g. Take
    3's step_14, genuinely "done out of checklist order" per
    annotations_master.csv) but resuming one performed dozens of steps ago
    is a real error."""
    if order_dist == 0:
        return 0.0
    if order_dist > 0:
        return -forward_rate * order_dist
    return -backward_rate * (-order_dist)


def _penalty_matrix(M: int, forward_rate: float, backward_rate: float) -> np.ndarray:
    idx = np.arange(M)
    order_dist = idx[:, None] - idx[None, :]  # [c, c'] = c - c'
    return np.where(
        order_dist > 0, -forward_rate * order_dist,
        np.where(order_dist < 0, -backward_rate * (-order_dist), 0.0),
    )


def viterbi_decode(
    log_emission: np.ndarray,
    duration_bounds: List[Tuple[float, float]],
    forward_rate: float = DEFAULT_FORWARD_RATE,
    backward_rate: float = DEFAULT_BACKWARD_RATE,
) -> List[int]:
    """Constraint-aware Viterbi (CAD, arXiv:2605.10149 section 3.2) adapted
    to our setting: valid start/end sets are the singleton first/last
    meta-state (see module docstring), and the Conf(A->B) frequency term is
    replaced by transition_penalty() (soft, order-distance based). Duration
    min/max gating follows the paper's forward-pass logic exactly. Fully
    vectorized over meta-states per frame -- O(M^2) per frame, O(M^2 * T)
    total, fast enough at our scale (M ~ 50, T ~ 10-15k frames) without a
    Python-level inner loop over states."""
    T, M = log_emission.shape
    if T == 0:
        return []

    d_min = np.array([b[0] for b in duration_bounds])
    d_max = np.array([b[1] for b in duration_bounds])
    penalty = _penalty_matrix(M, forward_rate, backward_rate)
    np.fill_diagonal(penalty, NEG_INF)  # self case handled separately below

    V = np.full((T, M), NEG_INF)
    D = np.zeros((T, M), dtype=np.int64)
    B = np.full((T, M), -1, dtype=np.int64)

    start_state, end_state = 0, M - 1
    V[0, start_state] = log_emission[0, start_state]
    D[0, start_state] = 1

    idx = np.arange(M)
    for t in range(1, T):
        prev_V, prev_D = V[t - 1], D[t - 1]
        reachable = prev_V > NEG_INF
        prev_frac = prev_D / T

        # transition-in candidates: cand[c, c'] = V[t-1,c'] + penalty[c,c']
        eligible_from = reachable & (prev_frac >= d_min)
        cand = prev_V[None, :] + penalty
        cand[:, ~eligible_from] = NEG_INF
        best_trans_from = np.argmax(cand, axis=1)
        best_trans_score = cand[idx, best_trans_from] + log_emission[t]

        # self-transition candidates
        self_ok = reachable & (prev_frac < d_max)
        self_score = np.where(self_ok, prev_V + log_emission[t], NEG_INF)

        use_self = self_score >= best_trans_score
        V[t] = np.where(use_self, self_score, best_trans_score)
        B[t] = np.where(use_self, idx, best_trans_from)
        D[t] = np.where(use_self, prev_D + 1, 1)

    if V[T - 1, end_state] <= NEG_INF:
        raise RuntimeError(
            "Viterbi failed to reach the end meta-state -- duration bounds are "
            "likely too tight for this take's actual timing. Widen shrink/grow "
            "in estimate_duration_bounds() and retry."
        )

    path = [0] * T
    path[T - 1] = end_state
    for t in range(T - 2, -1, -1):
        path[t] = int(B[t + 1, path[t + 1]])
    return path


# ==========================================
# Path -> segments
# ==========================================
@dataclass
class Segment:
    step: str
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def path_to_segments(path: List[int], meta_states: List[MetaState], fps: float = phase0.FPS) -> List[Segment]:
    """Collapses the per-frame meta-state path into contiguous runs and
    expands each run to a Segment per member step (>1 for a merged
    CO_LOCATED_GROUPS group). A step can produce more than one Segment if
    Viterbi legitimately revisits its meta-state later (e.g. step_28_2's two
    separate windows in Take 3's own ground truth) -- not collapsed, since
    that's real behavior, not an artifact."""
    segments: List[Segment] = []
    T = len(path)
    t = 0
    while t < T:
        c = path[t]
        start = t
        while t < T and path[t] == c:
            t += 1
        end = t - 1
        for step in meta_states[c].members:
            segments.append(Segment(
                step=step, start_frame=start, end_frame=end,
                start_sec=start / fps, end_sec=(end + 1) / fps,
            ))
    return segments


# ==========================================
# Segments -> frame-level multi-label active-step sets (VLM routing input)
# ==========================================
def segments_to_frame_labels(segments: List[Segment], total_frames: int) -> List[Set[str]]:
    """active_steps[t] = set of step ids active at frame t. Trivial by
    construction -- no extra smoothing/min-duration/concurrency
    post-processing needed on top: min-duration is already enforced by
    Viterbi's d_min, and concurrency (co-located groups) is already
    guaranteed by the meta-state merge, not something to re-enforce here."""
    labels: List[Set[str]] = [set() for _ in range(total_frames)]
    for seg in segments:
        lo, hi = max(seg.start_frame, 0), min(seg.end_frame, total_frames - 1)
        for t in range(lo, hi + 1):
            labels[t].add(seg.step)
    return labels


def gt_frame_multilabels(annotations_csv_path: str, total_frames: int, fps: float = phase0.FPS) -> List[Set[str]]:
    """Ground-truth active_steps(t), read directly from annotations_master.csv
    rows (which already list simultaneous steps as separate rows sharing a
    timestamp) -- unlike gt_frame_labels() below, this allows genuine
    multi-membership per frame rather than collapsing to one meta-label."""
    df = performed_rows(read_annotations(annotations_csv_path))
    labels: List[Set[str]] = [set() for _ in range(total_frames)]
    for _, row in df.iterrows():
        step = str(row["step_names"]).strip()
        fs = int(round(float(row["start_time"]) * fps))
        fe = int(round(float(row["end_time"]) * fps))
        for fr in range(max(fs, 0), min(fe, total_frames - 1) + 1):
            labels[fr].add(step)
    return labels


def multilabel_frame_metrics(pred: List[Set[str]], gt: List[Set[str]]) -> dict:
    """Per-frame multi-label agreement between predicted and ground-truth
    active_steps(t): mean Jaccard, mean precision/recall, exact-set-match
    rate, and the top steps responsible for false positives/negatives (which
    steps are most often over- or under-predicted) -- the confusion report
    needed to diagnose whether errors cluster on specific hard-to-place
    steps (e.g. the unanchored 25_1-25_4 cluster) or are spread evenly."""
    n = min(len(pred), len(gt))
    jaccards: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    exact = 0
    fp_counts: Dict[str, int] = {}
    fn_counts: Dict[str, int] = {}

    for i in range(n):
        p, g = pred[i], gt[i]
        inter = len(p & g)
        union = len(p | g)
        jaccards.append(inter / union if union else 1.0)
        precisions.append(inter / len(p) if p else (1.0 if not g else 0.0))
        recalls.append(inter / len(g) if g else (1.0 if not p else 0.0))
        if p == g:
            exact += 1
        for step in p - g:
            fp_counts[step] = fp_counts.get(step, 0) + 1
        for step in g - p:
            fn_counts[step] = fn_counts.get(step, 0) + 1

    return {
        "mean_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
        "mean_precision": float(np.mean(precisions)) if precisions else 0.0,
        "mean_recall": float(np.mean(recalls)) if recalls else 0.0,
        "exact_match_rate": exact / n if n else 0.0,
        "top_false_positive_steps": sorted(fp_counts.items(), key=lambda kv: -kv[1])[:10],
        "top_false_negative_steps": sorted(fn_counts.items(), key=lambda kv: -kv[1])[:10],
    }


# ==========================================
# Segment/interval-level metrics (IoU, F1@k, edit distance, frame accuracy)
# ==========================================
def segments_to_intervals(segments: List[Segment]) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for seg in segments:
        out.setdefault(seg.step, []).append((seg.start_sec, seg.end_sec))
    return out


def load_ground_truth_intervals(annotations_csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    df = performed_rows(read_annotations(annotations_csv_path))
    out: Dict[str, List[Tuple[float, float]]] = {}
    for _, row in df.iterrows():
        step = str(row["step_names"]).strip()
        out.setdefault(step, []).append((float(row["start_time"]), float(row["end_time"])))
    return out


def _interval_union_len(intervals: List[Tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def _interval_intersection_len(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> float:
    total = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            total += max(0.0, min(e1, e2) - max(s1, s2))
    return total


def iou_per_step(
    pred: Dict[str, List[Tuple[float, float]]],
    gt: Dict[str, List[Tuple[float, float]]],
) -> Dict[str, float]:
    """Interval-set IoU per step (handles multi-occurrence steps by union)."""
    ious: Dict[str, float] = {}
    for step in set(pred) | set(gt):
        p, g = pred.get(step, []), gt.get(step, [])
        inter = _interval_intersection_len(p, g)
        union = _interval_union_len(p) + _interval_union_len(g) - inter
        ious[step] = inter / union if union > 0 else 0.0
    return ious


def mean_start_error(
    pred: Dict[str, List[Tuple[float, float]]],
    gt: Dict[str, List[Tuple[float, float]]],
) -> float:
    errors = []
    for step, g in gt.items():
        p = pred.get(step)
        if not p:
            continue
        errors.append(abs(min(iv[0] for iv in p) - min(iv[0] for iv in g)))
    return float(np.mean(errors)) if errors else float("nan")


def _collapse_to_segments(label_sequence: List[str]) -> List[Tuple[str, int, int]]:
    segs: List[Tuple[str, int, int]] = []
    i, n = 0, len(label_sequence)
    while i < n:
        label = label_sequence[i]
        j = i
        while j < n and label_sequence[j] == label:
            j += 1
        segs.append((label, i, j - 1))
        i = j
    return segs


def f1_at_k(pred_segs: List[Tuple[str, int, int]], gt_segs: List[Tuple[str, int, int]], overlap: float) -> float:
    """Standard TAS F1@overlap (Lea et al. 2017 convention): greedy
    one-to-one IoU matching per predicted segment against unused
    same-label GT segments."""
    matched_gt = [False] * len(gt_segs)
    tp = 0
    for p_label, p_s, p_e in pred_segs:
        best_iou, best_j = 0.0, -1
        for j, (g_label, g_s, g_e) in enumerate(gt_segs):
            if matched_gt[j] or g_label != p_label:
                continue
            inter = max(0, min(p_e, g_e) - max(p_s, g_s) + 1)
            union = max(p_e, g_e) - min(p_s, g_s) + 1
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= overlap:
            matched_gt[best_j] = True
            tp += 1
    precision = tp / len(pred_segs) if pred_segs else 0.0
    recall = tp / len(gt_segs) if gt_segs else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall) * 100


def edit_score(pred_labels: List[str], gt_labels: List[str]) -> float:
    """Normalized Levenshtein distance over collapsed segment-label
    sequences: (1 - dist/max_len) * 100."""
    n, m = len(pred_labels), len(gt_labels)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if pred_labels[i - 1] == gt_labels[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[n][m]
    max_len = max(n, m)
    return (1 - dist / max_len) * 100 if max_len > 0 else 100.0


def frame_accuracy(pred_frame_labels: List[str], gt_frame_labels: List[str]) -> float:
    n = min(len(pred_frame_labels), len(gt_frame_labels))
    if n == 0:
        return 0.0
    correct = sum(1 for i in range(n) if pred_frame_labels[i] == gt_frame_labels[i])
    return correct / n * 100


def gt_frame_labels(
    annotations_csv_path: str,
    meta_states: List[MetaState],
    total_frames: int,
    fps: float = phase0.FPS,
) -> List[str]:
    """Per-frame GT labels at meta-state (single-label) granularity -- for
    frame_accuracy/F1/edit-distance, which need one label per frame; use
    gt_frame_multilabels() instead for the real multi-label VLM-routing
    comparison. Co-located groups collapse to one label here so
    single-label frame accuracy doesn't penalize the decoder for not
    picking which of several simultaneous steps is 'the' frame label."""
    df = performed_rows(read_annotations(annotations_csv_path))
    labels = ["__none__"] * total_frames
    for _, row in df.iterrows():
        step = str(row["step_names"]).strip()
        label = meta_label_for_step(step, meta_states)
        fs = int(round(float(row["start_time"]) * fps))
        fe = int(round(float(row["end_time"]) * fps))
        for fr in range(max(fs, 0), min(fe + 1, total_frames)):
            labels[fr] = label
    return labels


def pred_frame_labels(path: List[int], meta_states: List[MetaState]) -> List[str]:
    return [meta_states[c].label for c in path]


def compute_metrics(
    path: List[int],
    meta_states: List[MetaState],
    annotations_csv_path: str,
    total_frames: int,
    fps: float = phase0.FPS,
) -> dict:
    segments = path_to_segments(path, meta_states, fps)
    pred_intervals = segments_to_intervals(segments)
    gt_intervals = load_ground_truth_intervals(annotations_csv_path)

    pred_labels_f = pred_frame_labels(path, meta_states)
    gt_labels_f = gt_frame_labels(annotations_csv_path, meta_states, total_frames, fps)

    pred_segs = _collapse_to_segments(pred_labels_f)
    gt_segs = _collapse_to_segments(gt_labels_f)

    ious = iou_per_step(pred_intervals, gt_intervals)

    pred_multilabels = segments_to_frame_labels(segments, total_frames)
    gt_multilabels = gt_frame_multilabels(annotations_csv_path, total_frames, fps)

    return {
        "iou_per_step": ious,
        "mean_iou": float(np.mean(list(ious.values()))) if ious else 0.0,
        "mean_start_error_sec": mean_start_error(pred_intervals, gt_intervals),
        "f1_at_10": f1_at_k(pred_segs, gt_segs, 0.10),
        "f1_at_25": f1_at_k(pred_segs, gt_segs, 0.25),
        "f1_at_50": f1_at_k(pred_segs, gt_segs, 0.50),
        "edit_score": edit_score([s[0] for s in pred_segs], [s[0] for s in gt_segs]),
        "frame_accuracy": frame_accuracy(pred_labels_f, gt_labels_f),
        "multilabel": multilabel_frame_metrics(pred_multilabels, gt_multilabels),
    }


# ==========================================
# Orchestration
# ==========================================
def _setup_logging(log_path: str) -> logging.Logger:
    log = logging.getLogger("segmentation")
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


def save_results(paths: dict, segments: List[Segment], metrics: Optional[dict]) -> None:
    out = {
        "segments": [dataclasses.asdict(s) for s in segments],
        "metrics": metrics,
    }
    with open(paths["segments_json"], "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def run_segmentation(
    take_id: int,
    embed_backend: str = "auto",
    duration_reference_take_id: Optional[int] = None,
    sigma_sec: float = DEFAULT_SIGMA_SEC,
    forward_rate: float = DEFAULT_FORWARD_RATE,
    backward_rate: float = DEFAULT_BACKWARD_RATE,
    shrink: float = DEFAULT_DURATION_SHRINK,
    grow: float = DEFAULT_DURATION_GROW,
    zone_weight: float = DEFAULT_ZONE_WEIGHT,
    evaluate: bool = True,
) -> Tuple[List[Segment], Optional[dict]]:
    """Decodes step boundaries for `take_id` from its transcript + wide
    keypoints CSV, using duration bounds fit from
    `duration_reference_take_id`'s annotations_master.csv (defaults to the
    same take -- see module docstring's circularity caveat). Requires
    phase0's outputs (anchor_embeddings.npy/anchor_index.json) and
    features.py's wide_keypoints_csv to already exist for `take_id`. Writes
    segments.json to the take's processed folder and returns
    (segments, metrics)."""
    paths = config.get_take_paths(take_id)
    logger = _setup_logging(paths["segmentation_log"])
    logger.info(f"── Segmentation  [Take {take_id}] ──────────────")

    meta_states = build_meta_states()
    wide_df = pd.read_csv(paths["wide_keypoints_csv"])
    total_frames = int(wide_df["Frame"].max()) + 1
    total_duration_sec = total_frames / phase0.FPS
    logger.info(f"  {total_frames} frames  ({total_duration_sec:.1f}s @ {phase0.FPS} fps)  "
                f"{len(meta_states)} meta-states (from 56 steps)")

    utterances = parse_transcript(paths["transcript"])
    logger.info(f"  Parsed {len(utterances)} transcript utterances")

    backend = phase0.EmbedBackend(embed_backend, logger=logger)
    anchor_vectors, anchor_index = load_step_anchors(paths["anchor_embeddings"], paths["anchor_index"])
    matches = match_anchors_to_transcript(anchor_vectors, anchor_index, utterances, backend)
    logger.info(f"  {len(matches)}/{len(anchor_index)} step-anchor phrases matched above sim floor")

    raw_emission = build_emission_matrix(meta_states, matches, total_frames, sigma_sec=sigma_sec)
    if zone_weight:
        # Off by default -- see DEFAULT_ZONE_WEIGHT; skipped entirely rather
        # than multiplied by zero so the pose scan isn't paid for when unused.
        raw_emission = raw_emission + zone_weight * build_zone_emission(meta_states, wide_df)
    logger.info(f"  Emission: transcript anchors (sigma={sigma_sec}s)"
                + (f" + pose zones (weight={zone_weight})" if zone_weight else "; zone signal OFF"))
    n_flat = int((raw_emission.max(axis=0) <= 0).sum())
    if n_flat:
        logger.warning(
            f"  {n_flat}/{len(meta_states)} meta-states have NO emission evidence at all; "
            f"whether they appear in the output is decided by tie-breaking, not by data."
        )
    log_emission = emission_log_probs(raw_emission)

    ref_take = take_id if duration_reference_take_id is None else duration_reference_take_id
    if ref_take == take_id:
        logger.warning(
            "  Duration bounds fit from the SAME take being decoded -- circular "
            "(fit-then-eval). Pass duration_reference_take_id pointing at a "
            "different annotated take once one exists."
        )
    ref_paths = config.get_take_paths(ref_take)
    duration_bounds = estimate_duration_bounds(
        ref_paths["annotations_csv"], meta_states, total_duration_sec, shrink=shrink, grow=grow,
    )

    path = viterbi_decode(log_emission, duration_bounds, forward_rate=forward_rate, backward_rate=backward_rate)
    segments = path_to_segments(path, meta_states)
    logger.info(f"  Decoded {len(segments)} step segments")

    metrics = None
    if evaluate and os.path.exists(paths["annotations_csv"]):
        metrics = compute_metrics(path, meta_states, paths["annotations_csv"], total_frames)
        ml = metrics["multilabel"]
        logger.info(
            f"  mean_iou={metrics['mean_iou']:.3f}  "
            f"F1@{{10,25,50}}={metrics['f1_at_10']:.1f}/{metrics['f1_at_25']:.1f}/{metrics['f1_at_50']:.1f}  "
            f"edit={metrics['edit_score']:.1f}  frame_acc={metrics['frame_accuracy']:.1f}  "
            f"start_err={metrics['mean_start_error_sec']:.2f}s"
        )
        logger.info(
            f"  [multilabel] jaccard={ml['mean_jaccard']:.3f}  precision={ml['mean_precision']:.3f}  "
            f"recall={ml['mean_recall']:.3f}  exact_match={ml['exact_match_rate']:.3f}"
        )

    save_results(paths, segments, metrics)
    return segments, metrics
