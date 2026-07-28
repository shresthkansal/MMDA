"""
Person re-identification / Doctor-Patient role assignment from raw YOLO tracking output.

Ported from FS_model.ipynb. Two independent approaches exist in the source
notebook for the same problem (assign canonical Doctor/Patient identities
to YOLO's raw, unstable per-frame track IDs):

1. `SimpleReIDTracker` / `reassign_ids_simple` / `assign_roles_by_behavior`
   (cell idx 19) -- fully automatic, feature-based (position + skeleton
   proportions + movement heuristics). No confirmed end-to-end successful
   run exists anywhere in the notebook's cached output for this approach.

2. The "ID-based tracker" chain (cells idx 29-31) -- semi-supervised: scan
   for ID-change phases, a human visually verifies doctor/patient identity
   at each phase, then a strict anchor-schedule tracker propagates those
   verified identities across frames. THIS is the approach with confirmed
   real successful output (idx 31: "Tracking Complete. Saved to:
   .../ReID_Results/...", and idx 32's batch EDA shows >98% doctor/patient
   visibility for Takes 2, 3, 5, 6 processed this way).

Both are ported (per "keep unique working code, judge later"), but #2 is
the one to actually use -- #1 is included for reference/comparison only.
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Approach 1: automatic feature-based tracker (UNCONFIRMED end-to-end -- reference only)
# ==========================================
class SimpleReIDTracker:
    """
    Simple tracker using position and skeleton features.
    Assumes: Patient stays on bed (low Y), Doctor moves around (higher Y, more movement).
    """

    def __init__(self):
        self.tracks = {}  # {canonical_id: {'positions': [], 'features': [], 'last_frame': int}}
        self.next_id = 1

    def _get_position_features(self, kp_row):
        positions = []
        for kp in ["L_Shoulder", "R_Shoulder", "L_Hip", "R_Hip"]:
            x = kp_row.get(f"{kp}_X", 0)
            y = kp_row.get(f"{kp}_Y", 0)
            if x > 0 and y > 0:
                positions.append([x, y])

        if len(positions) < 2:
            return None

        positions = np.array(positions)
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        y_min = positions[:, 0].min()
        y_max = positions[:, 1].max()
        vertical_extent = y_max - y_min

        return {"center_x": center_x, "center_y": center_y, "vertical_extent": vertical_extent}

    def _get_skeleton_features(self, kp_row):
        l_sh = np.array([kp_row.get("L_Shoulder_X", 0), kp_row.get("L_Shoulder_Y", 0)])
        r_sh = np.array([kp_row.get("R_Shoulder_X", 0), kp_row.get("R_Shoulder_Y", 0)])
        l_hp = np.array([kp_row.get("L_Hip_X", 0), kp_row.get("L_Hip_Y", 0)])
        r_hp = np.array([kp_row.get("R_Hip_X", 0), kp_row.get("R_Hip_Y", 0)])

        if any(p[0] <= 0 for p in [l_sh, r_sh, l_hp, r_hp]):
            return None

        shoulder_width = np.linalg.norm(r_sh - l_sh)
        hip_width = np.linalg.norm(r_hp - l_hp)
        torso_height = np.linalg.norm((l_sh + r_sh) / 2 - (l_hp + r_hp) / 2)

        if torso_height < 1:
            return None

        return np.array([
            shoulder_width / torso_height,
            hip_width / torso_height,
            shoulder_width / (hip_width + 1e-6),
        ])

    def match_detection(self, frame, yolo_id, kp_row, position_weight=0.6, skeleton_weight=0.4, max_gap=150):
        pos_feat = self._get_position_features(kp_row)
        skel_feat = self._get_skeleton_features(kp_row)

        if pos_feat is None or skel_feat is None:
            return -1

        best_match_id = None
        best_score = 0

        for track_id, track_data in self.tracks.items():
            if frame - track_data["last_frame"] > max_gap:
                continue

            recent_positions = track_data["positions"][-20:]
            recent_features = track_data["features"][-20:]

            if len(recent_positions) == 0:
                continue

            recent_centers = np.array([[p["center_x"], p["center_y"]] for p in recent_positions])
            current_center = np.array([pos_feat["center_x"], pos_feat["center_y"]])

            pos_distances = np.linalg.norm(recent_centers - current_center, axis=1)
            avg_pos_distance = np.mean(pos_distances)
            pos_score = np.exp(-avg_pos_distance / 100)

            skel_similarities = [cosine_similarity([skel_feat], [f])[0][0] for f in recent_features]
            avg_skel_sim = np.mean(skel_similarities)

            total_score = position_weight * pos_score + skeleton_weight * avg_skel_sim

            if total_score > best_score:
                best_score = total_score
                best_match_id = track_id

        threshold = 0.5

        if best_score > threshold and best_match_id is not None:
            canonical_id = best_match_id
        else:
            canonical_id = self.next_id
            self.next_id += 1
            self.tracks[canonical_id] = {"positions": [], "features": [], "last_frame": frame}

        self.tracks[canonical_id]["positions"].append(pos_feat)
        self.tracks[canonical_id]["features"].append(skel_feat)
        self.tracks[canonical_id]["last_frame"] = frame

        return canonical_id


def reassign_ids_simple(yolo_csv_path: str, output_csv_path: str):
    """Automatic ReID without face matching. See module docstring: unconfirmed end-to-end."""
    df = pd.read_csv(yolo_csv_path)
    tracker = SimpleReIDTracker()

    canonical_ids = []
    for idx, row in df.iterrows():
        frame = row["Frame"]
        canonical_id = tracker.match_detection(frame, row["Person_ID"], row)
        canonical_ids.append(canonical_id)
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(df)}")

    df["Canonical_ID"] = canonical_ids
    df = df[df["Canonical_ID"] != -1]
    df.to_csv(output_csv_path, index=False)

    print(f"\nCanonical IDs found: {sorted(df['Canonical_ID'].unique())}")
    print("\nID distribution:")
    print(df["Canonical_ID"].value_counts())

    return tracker, df


def assign_roles_by_behavior(reid_csv_path: str, output_csv_path: str) -> dict:
    """Assign Doctor/Patient based on movement and position patterns."""
    df = pd.read_csv(reid_csv_path)
    role_assignment = {}

    for canonical_id in df["Canonical_ID"].unique():
        person_df = df[df["Canonical_ID"] == canonical_id]

        positions = []
        for _, row in person_df.iterrows():
            shoulder_y = (row.get("L_Shoulder_Y", 0) + row.get("R_Shoulder_Y", 0)) / 2
            shoulder_x = (row.get("L_Shoulder_X", 0) + row.get("R_Shoulder_X", 0)) / 2
            if shoulder_y > 0:
                positions.append([shoulder_x, shoulder_y])

        if len(positions) < 10:
            role_assignment[canonical_id] = "Unknown"
            continue

        positions = np.array(positions)
        movement_score = np.std(positions, axis=0).mean()
        avg_y = np.mean(positions[:, 1])

        print(f"Canonical_ID {canonical_id}:")
        print(f"  Movement score: {movement_score:.1f}")
        print(f"  Average Y position: {avg_y:.1f}")
        print(f"  Frames present: {len(person_df)}")

        role_assignment[canonical_id] = {"movement": movement_score, "avg_y": avg_y, "frames": len(person_df)}

    sorted_by_movement = sorted(
        role_assignment.items(),
        key=lambda x: x[1]["movement"] if isinstance(x[1], dict) else 0,
        reverse=True,
    )

    if len(sorted_by_movement) >= 2:
        doctor_id = sorted_by_movement[0][0]
        patient_id = sorted_by_movement[1][0]
        final_roles = {doctor_id: "Doctor", patient_id: "Patient"}
    else:
        final_roles = {sorted_by_movement[0][0]: "Unknown"}

    print("\n=== Role Assignment ===")
    for cid, role in final_roles.items():
        print(f"Canonical_ID {cid} -> {role}")

    df["Role"] = df["Canonical_ID"].map(final_roles)
    df.to_csv(output_csv_path, index=False)

    return final_roles


# ==========================================
# Approach 2: semi-supervised anchor-schedule tracker (CONFIRMED WORKING -- use this one)
# ==========================================
def auto_generate_known_phases(csv_path: str, start_offset: int = 0, scan_interval: int = 100) -> dict:
    """Step 1: scan the raw YOLO CSV for candidate ID-change phases.

    Returns a dict for a human to visually spot-check (e.g. via
    render_phase_check_frames below) before feeding into build_anchor_schedule.
    """
    df = pd.read_csv(csv_path)
    total_f = int(df["Frame"].max())

    timeline = []
    scan_frames = list(range(start_offset, total_f, scan_interval))
    if total_f > 50 and (total_f - 50) > scan_frames[-1]:
        scan_frames.append(total_f - 50)

    for f in scan_frames:
        window = df[(df["Frame"] >= f) & (df["Frame"] < f + 50)]
        if window.empty:
            continue

        counts = window["Person_ID"].value_counts()
        valid_ids = sorted([pid for pid, c in counts.items() if c > 5])
        if not valid_ids:
            continue

        # Find the frame where the last person in this group actually arrives.
        first_appearances = [window[window["Person_ID"] == vid]["Frame"].min() for vid in valid_ids]
        actual_frame = int(max(first_appearances))

        timeline.append({"frame": actual_frame, "ids": valid_ids})

    phases = {}
    if not timeline:
        return {}

    current_ids = timeline[0]["ids"]
    prev_pat_guess = current_ids[0] if len(current_ids) > 0 else 0

    phases["Phase 0 (Start)"] = {
        "approx_frame": timeline[0]["frame"],
        "pat_id": float(prev_pat_guess),
        "doc_id": float(current_ids[1]) if len(current_ids) > 1 else 0.0,
    }

    phase_idx = 1
    for entry in timeline[1:]:
        ids = entry["ids"]
        if ids != current_ids and len(ids) > 0:
            if prev_pat_guess in ids:
                pat_id = prev_pat_guess
                others = [x for x in ids if x != pat_id]
                doc_id = others[0] if others else 0
            else:
                pat_id = ids[0]
                doc_id = ids[1] if len(ids) > 1 else 0
                prev_pat_guess = pat_id

            phases[f"Phase {phase_idx}"] = {
                "approx_frame": entry["frame"],
                "doc_id": float(doc_id),
                "pat_id": float(pat_id),
            }
            current_ids = ids
            phase_idx += 1

    last_entry = timeline[-1]
    last_ids = last_entry["ids"]

    if prev_pat_guess in last_ids:
        end_pat_id = prev_pat_guess
        others = [x for x in last_ids if x != end_pat_id]
        end_doc_id = others[0] if others else 0
    else:
        end_pat_id = last_ids[0]
        end_doc_id = last_ids[1] if len(last_ids) > 1 else 0

    last_recorded_frame = phases[list(phases.keys())[-1]]["approx_frame"]
    if abs(last_entry["frame"] - last_recorded_frame) > 100:
        phases[f"Phase {phase_idx} (End)"] = {
            "approx_frame": last_entry["frame"],
            "doc_id": float(end_doc_id),
            "pat_id": float(end_pat_id),
        }

    return phases


def find_true_start(pid: float, approx_f: int, dataframe: pd.DataFrame) -> int:
    """Finds the start of the specific tracking segment containing approx_f."""
    if pid == 0:
        return approx_f

    p_frames = np.sort(dataframe[dataframe["Person_ID"] == pid]["Frame"].unique())
    if len(p_frames) == 0:
        return approx_f

    dists = np.abs(p_frames - approx_f)
    idx_closest = np.argmin(dists)
    if dists[idx_closest] > 500:
        return approx_f

    start_frame = p_frames[idx_closest]
    for i in range(idx_closest, 0, -1):
        curr_f = p_frames[i]
        prev_f = p_frames[i - 1]
        if (curr_f - prev_f) > 60:
            start_frame = curr_f
            break
        start_frame = prev_f

    return int(start_frame)


def _verify_presence_and_add(df: pd.DataFrame, target_frame: int, intended_doc: float,
                              intended_pat: float, container: dict) -> None:
    """Strictly checks the CSV: is intended_doc/intended_pat actually at target_frame?"""
    actual_doc = 0.0
    if intended_doc > 0:
        exists = not df[(df["Frame"] == target_frame) & (df["Person_ID"] == intended_doc)].empty
        if exists:
            actual_doc = intended_doc

    actual_pat = 0.0
    if intended_pat > 0:
        exists = not df[(df["Frame"] == target_frame) & (df["Person_ID"] == intended_pat)].empty
        if exists:
            actual_pat = intended_pat

    if target_frame not in container:
        container[target_frame] = {"Doctor": actual_doc, "Patient": actual_pat}
    else:
        if actual_doc > 0:
            container[target_frame]["Doctor"] = actual_doc
        if actual_pat > 0:
            container[target_frame]["Patient"] = actual_pat


def build_anchor_schedule(known_phases: dict, csv_path: str) -> dict:
    """Step 2: turn human-verified KNOWN_PHASES into a strict frame->{Doctor,Patient} anchor schedule.

    `known_phases` is the (optionally human-edited) output of
    auto_generate_known_phases. Returns ANCHOR_IDS, ready for
    run_id_based_tracker.
    """
    df = pd.read_csv(csv_path)
    anchor_ids: dict = {}

    for name, data in known_phases.items():
        approx = data["approx_frame"]
        doc_id = data["doc_id"]
        pat_id = data["pat_id"]

        start_doc = find_true_start(doc_id, approx, df)
        start_pat = find_true_start(pat_id, approx, df)

        _verify_presence_and_add(df, start_doc, doc_id, pat_id, anchor_ids)
        if abs(start_pat - start_doc) > 10:
            _verify_presence_and_add(df, start_pat, doc_id, pat_id, anchor_ids)

    return dict(sorted(anchor_ids.items()))


def render_phase_check_frames(video_path: str, csv_path: str, phases: dict) -> None:
    """Optional human sanity-check: draws doc/patient circles at each phase's frame.

    Not part of the automated pipeline -- call manually from a Colab cell
    (uses cv2_imshow) if you want to eyeball a schedule before trusting it.
    """
    from google.colab.patches import cv2_imshow

    cap = cv2.VideoCapture(video_path)
    df = pd.read_csv(csv_path)

    for name, data in phases.items():
        t_frame = int(data["approx_frame"])
        doc_id, pat_id = data.get("doc_id", data.get("Doctor", 0)), data.get("pat_id", data.get("Patient", 0))

        cap.set(cv2.CAP_PROP_POS_FRAMES, t_frame)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {t_frame} for {name}")
            continue

        frame_data = df[df["Frame"] == t_frame]

        for pid, tag, color in [(doc_id, "DOC", (0, 255, 0)), (pat_id, "PAT", (0, 0, 255))]:
            row = frame_data[frame_data["Person_ID"] == pid]
            if not row.empty:
                r = row.iloc[0]
                kps = [[r[f"{k}_X"], r[f"{k}_Y"]] for k in ["L_Shoulder", "R_Shoulder", "L_Hip", "R_Hip"]
                       if r.get(f"{k}_X", 0) > 0]
                if kps:
                    c = np.mean(kps, axis=0)
                    cv2.circle(frame, (int(c[0]), int(c[1])), 20, color, 3)
                    cv2.putText(frame, f"{tag} {int(pid)}", (int(c[0]) + 25, int(c[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, f"{name} @ F{t_frame}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * 0.4), int(h * 0.4)))
        cv2_imshow(small)

    cap.release()


def run_id_based_tracker(input_csv: str, output_csv: str, anchor_ids: dict) -> pd.DataFrame:
    """Step 3: propagate the verified anchor schedule across all frames, labeling
    Doctor/Patient/Noise, and writing the final labeled CSV."""
    print(f"Starting ID-based tracking on {input_csv}...")
    df = pd.read_csv(input_csv)

    sorted_anchors = sorted(anchor_ids.keys())
    current_anchor_idx = 0

    active_doc_id = anchor_ids[sorted_anchors[0]]["Doctor"]
    active_pat_id = anchor_ids[sorted_anchors[0]]["Patient"]

    processed_rows = []

    for frame_idx, group in df.groupby("Frame"):
        while current_anchor_idx < len(sorted_anchors) and frame_idx >= sorted_anchors[current_anchor_idx]:
            anchor_frame = sorted_anchors[current_anchor_idx]
            active_doc_id = anchor_ids[anchor_frame]["Doctor"]
            active_pat_id = anchor_ids[anchor_frame]["Patient"]
            current_anchor_idx += 1

        for _, row in group.iterrows():
            pid = row["Person_ID"]
            new_row = row.copy()

            if pid == active_doc_id:
                new_row["Canonical_ID"] = 1
                new_row["Role"] = "Doctor"
            elif pid == active_pat_id:
                new_row["Canonical_ID"] = 2
                new_row["Role"] = "Patient"
            else:
                new_row["Canonical_ID"] = 3
                new_row["Role"] = "Noise"

            processed_rows.append(new_row)

    final_df = pd.DataFrame(processed_rows)
    final_df = final_df[final_df["Canonical_ID"].isin([1, 2])]
    final_df = final_df.sort_values(by=["Frame", "Canonical_ID"])
    final_df.to_csv(output_csv, index=False)
    print(f"Tracking complete. Saved to: {output_csv}")
    return final_df
