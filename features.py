"""
Final feature-engineering pipeline: the confirmed-good core of FS_model.ipynb.

Per the user: "the features that actually are worth exporting forward to the
next stage are everything after Final Dataset Preparation." Ported here with
the most care/fidelity, preserving exact computation logic from the source
notebook (idx 106-130), in original execution order:

    preprocess_medical_keypoints      (merge Front/Side/360 CSVs -> wide format)
    construct_binary_step_columns     (annotate step intervals -- SEE WARNING BELOW)
    add_smoothed_posture_features     (spine angle, shoulder tilt -> Shape/Twist)
    add_strict_touch_features         (proximity-based touch detection)
    add_doctor_position_feature       (Head/Middle/Foot position relative to patient)
    merge_llm_features                (Gemini transcript -> Audio_Umbrella_Prediction)
    add_velocity_features             (hand velocity, smoothed)
    add_anatomical_zones              (distance-to-chest/neck/wrists)
    update_master_dataset             (continuous geometric zone projection -- LATEST
                                        version, idx 130, supersedes the near-identical
                                        pair in idx 129 which was part of an excluded
                                        interactive ipywidgets visualizer)

*** IMPORTANT BUG FOUND, NOT SILENTLY FIXED ***
`construct_binary_step_columns` ran without error and is what actually
produced the existing Wide_Keypoints2.csv files on Drive, BUT its frame-to-
step mapping loop was commented out in the source notebook. As shipped, it
only *initializes* one binary column per step (all zeros) and never actually
marks which frames belong to which step. This is ported faithfully (matching
what actually ran and produced your existing data) with the dead loop
preserved in a comment and a runtime warning added so it can't be missed.
Re-enabling it is a real logic change, left for a deliberate decision rather
than done silently here.
"""
import os

import cv2
import numpy as np
import pandas as pd


# ==========================================
# 1. Merge Front/Side/360 CSVs -> wide format
# ==========================================
def preprocess_medical_keypoints(csv_paths: dict) -> "pd.DataFrame | None":
    """csv_paths: {'Front': path, 'Side': path, '360': path, ...}.
    Returns one wide-format row per Frame:
    [Frame, Front_Doctor_Nose_X, Front_Patient_Nose_X, Side_Doctor_..., ...]
    """
    master_df = None

    for view_name, filepath in csv_paths.items():
        if not os.path.exists(filepath):
            print(f"Warning: {view_name} file not found at {filepath}")
            continue

        print(f"Processing {view_name} view...")
        df = pd.read_csv(filepath)

        # COCO-style occlusion handling: (0,0) means "not detected" -> NaN
        coord_cols = [c for c in df.columns if c.endswith("_X") or c.endswith("_Y")]
        df[coord_cols] = df[coord_cols].replace(0, np.nan)
        df[coord_cols] = df[coord_cols].replace(0.0, np.nan)

        df_doc = df[df["Role"] == "Doctor"].copy()
        df_pat = df[df["Role"] == "Patient"].copy()

        drop_cols = ["Person_ID", "Canonical_ID", "Role"]
        doc_rename = {c: f"{view_name}_Doctor_{c}" for c in df_doc.columns if c not in ["Frame"] and c not in drop_cols}
        pat_rename = {c: f"{view_name}_Patient_{c}" for c in df_pat.columns if c not in ["Frame"] and c not in drop_cols}

        df_doc = df_doc.rename(columns=doc_rename).drop(columns=drop_cols, errors="ignore")
        df_pat = df_pat.rename(columns=pat_rename).drop(columns=drop_cols, errors="ignore")

        # Inner-join semantics preserved via outer merge on Frame (both must exist
        # for interaction features, but we don't drop frames where only one is present).
        df_view = pd.merge(df_doc, df_pat, on="Frame", how="outer")

        if master_df is None:
            master_df = df_view
        else:
            master_df = pd.merge(master_df, df_view, on="Frame", how="outer")

    if master_df is not None:
        master_df = master_df.sort_values("Frame").reset_index(drop=True)
        print(f"Success! Merged shape: {master_df.shape}")

    return master_df


# ==========================================
# 2. Binary step columns -- SEE MODULE WARNING, mapping loop is a documented no-op
# ==========================================
def construct_binary_step_columns(master_csv_path: str, annotations_csv_path: str, output_csv_path: str) -> None:
    """WARNING: as ported from the source notebook, this only initializes one
    binary column per step (all zeros) -- the frame-range mapping that should
    set them to 1 was commented out in the original and is preserved as a
    comment below, not executed. See module docstring."""
    print("WARNING: construct_binary_step_columns' frame-to-step mapping is "
          "disabled (ported as-is from source -- see features.py docstring). "
          "Output binary step columns will all be 0.")

    df_master = pd.read_csv(master_csv_path)
    df_ann = pd.read_csv(annotations_csv_path)

    df_master.columns = df_master.columns.str.strip().str.replace("\t", "")
    df_ann.columns = df_ann.columns.str.strip().str.replace("\t", "")

    print(f"Detected annotation columns: {list(df_ann.columns)}")

    name_col = None
    for possible_name in ["step_names", "step_name", "name", "step id"]:
        if possible_name in df_ann.columns:
            name_col = possible_name
            break
    if name_col is None:
        raise ValueError(f"Could not find a name column. Available columns are: {list(df_ann.columns)}")

    df_master["Frame"] = pd.to_numeric(df_master["Frame"], errors="coerce")
    df_ann["start_frame"] = pd.to_numeric(df_ann["start_frame"], errors="coerce")
    df_ann["end_frame"] = pd.to_numeric(df_ann["end_frame"], errors="coerce")

    unique_steps = df_ann[name_col].astype(str).str.strip().unique()
    for step in unique_steps:
        df_master[step] = 0

    # --- Original mapping logic (disabled in source, preserved verbatim) ---
    # frames = df_master['Frame'].values
    # for _, row in df_ann.iterrows():
    #     s_frame = row['start_frame']
    #     e_frame = row['end_frame']
    #     step_col = str(row[name_col]).strip()
    #     if pd.notna(s_frame) and pd.notna(e_frame):
    #         mask = (frames >= s_frame) & (frames <= e_frame)
    #         df_master.loc[mask, step_col] = 1

    df_master.to_csv(output_csv_path, index=False)
    print(f"Dataset saved to {output_csv_path}")


# ==========================================
# 3. Posture / touch / doctor-position features
# ==========================================
def calculate_posture_angles(row) -> pd.Series:
    dy_spine = ((row.get("Side_Patient_L_Hip_Y", 0) + row.get("Side_Patient_R_Hip_Y", 0)) / 2) - \
               ((row.get("Side_Patient_L_Shoulder_Y", 0) + row.get("Side_Patient_R_Shoulder_Y", 0)) / 2)
    dx_spine = ((row.get("Side_Patient_L_Shoulder_X", 0) + row.get("Side_Patient_R_Shoulder_X", 0)) / 2) - \
               ((row.get("Side_Patient_L_Hip_X", 0) + row.get("Side_Patient_R_Hip_X", 0)) / 2)
    spine_angle = np.degrees(np.arctan2(dy_spine, abs(dx_spine))) if pd.notna(dy_spine) else 0.0

    lsy = row.get("Front_Patient_L_Shoulder_Y", 0)
    rsy = row.get("Front_Patient_R_Shoulder_Y", 0)
    lsx = row.get("Front_Patient_L_Shoulder_X", 0)
    rsx = row.get("Front_Patient_R_Shoulder_X", 0)
    tilt = np.degrees(np.arctan2(lsy - rsy, abs(rsx - lsx))) if pd.notna(lsy) else 0.0

    return pd.Series([spine_angle, tilt])


def add_smoothed_posture_features(df: pd.DataFrame, window_size: int = 15) -> pd.DataFrame:
    df[["Spine_Angle_Raw", "Shoulder_Tilt_Raw"]] = df.apply(calculate_posture_angles, axis=1)

    df["Spine_Angle"] = df["Spine_Angle_Raw"].rolling(window=window_size, center=True, min_periods=1).median()
    df["Shoulder_Tilt"] = df["Shoulder_Tilt_Raw"].rolling(window=window_size, center=True, min_periods=1).median()

    df["Shape"] = np.where(df["Spine_Angle"] > 45, "Sitting Up", "Lying Down")
    df["Twist"] = np.select(
        [df["Shoulder_Tilt"] > 25, df["Shoulder_Tilt"] < -25],
        ["Left Lateral (Turned)", "Right Lateral (Turned)"],
        default="Supine (Flat)",
    )

    df.drop(columns=["Spine_Angle_Raw", "Shoulder_Tilt_Raw"], inplace=True)
    return df


PAT_BODY_PARTS = [
    "Nose", "Neck", "R_Shoulder", "L_Shoulder", "R_Elbow", "L_Elbow", "R_Wrist", "L_Wrist",
    "R_Hip", "L_Hip", "R_Knee", "L_Knee", "R_Ankle", "L_Ankle",
]


def add_strict_touch_features(df: pd.DataFrame, threshold: int = 75) -> pd.DataFrame:
    def calculate_strict_proximity(row):
        touches = {"L": False, "R": False}
        for hand in ["L", "R"]:
            dh = f"{hand}_Wrist"
            for pb in PAT_BODY_PARTS:
                fx_doc, fy_doc = row.get(f"Front_Doctor_{dh}_X"), row.get(f"Front_Doctor_{dh}_Y")
                fx_pat, fy_pat = row.get(f"Front_Patient_{pb}_X"), row.get(f"Front_Patient_{pb}_Y")
                sx_doc, sy_doc = row.get(f"Side_Doctor_{dh}_X"), row.get(f"Side_Doctor_{dh}_Y")
                sx_pat, sy_pat = row.get(f"Side_Patient_{pb}_X"), row.get(f"Side_Patient_{pb}_Y")

                if pd.isna([fx_doc, fy_doc, fx_pat, fy_pat, sx_doc, sy_doc, sx_pat, sy_pat]).any():
                    continue

                dist_front = np.sqrt((fx_doc - fx_pat) ** 2 + (fy_doc - fy_pat) ** 2)
                dist_side = np.sqrt((sx_doc - sx_pat) ** 2 + (sy_doc - sy_pat) ** 2)

                if dist_front < threshold and dist_side < threshold:
                    touches[hand] = True
                    break
        return pd.Series([touches["L"], touches["R"]])

    df[["L_Is_Touching", "R_Is_Touching"]] = df.apply(calculate_strict_proximity, axis=1)

    df["L_Is_Touching"] = df["L_Is_Touching"].rolling(window=15, center=True, min_periods=1).apply(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0], raw=False).astype(bool)
    df["R_Is_Touching"] = df["R_Is_Touching"].rolling(window=15, center=True, min_periods=1).apply(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0], raw=False).astype(bool)
    return df


def add_doctor_position_feature(df: pd.DataFrame, window_size: int = 15) -> pd.DataFrame:
    def calculate_position(row):
        doc_hx = row.get("Side_Doctor_L_Hip_X")
        doc_hy = row.get("Side_Doctor_R_Hip_X")
        if pd.isna(doc_hx) and pd.isna(doc_hy):
            return "Unknown"
        doc_x = np.nanmean([doc_hx, doc_hy])

        pat_sh_x = np.nanmean([row.get("Side_Patient_L_Shoulder_X"), row.get("Side_Patient_R_Shoulder_X")])
        pat_knee_x = np.nanmean([row.get("Side_Patient_L_Knee_X"), row.get("Side_Patient_R_Knee_X")])

        if np.isnan(pat_sh_x) or np.isnan(pat_knee_x):
            return "Unknown"

        head_on_left = pat_sh_x < pat_knee_x

        if head_on_left:
            if doc_x < pat_sh_x:
                return "Head"
            elif doc_x > pat_knee_x:
                return "Foot"
            else:
                return "Middle"
        else:
            if doc_x > pat_sh_x:
                return "Head"
            elif doc_x < pat_knee_x:
                return "Foot"
            else:
                return "Middle"

    df["Position_Raw"] = df.apply(calculate_position, axis=1)

    unique_pos = df["Position_Raw"].unique()
    pos_to_id = {pos: i for i, pos in enumerate(unique_pos)}
    id_to_pos = {i: pos for pos, i in pos_to_id.items()}

    df["Pos_ID"] = df["Position_Raw"].map(pos_to_id)
    df["Pos_ID_Smoothed"] = (
        df["Pos_ID"].rolling(window=window_size, center=True, min_periods=1).median().bfill().ffill().round().astype(int)
    )

    df["Doctor_Position"] = df["Pos_ID_Smoothed"].map(id_to_pos)
    df.drop(columns=["Position_Raw", "Pos_ID", "Pos_ID_Smoothed"], inplace=True)
    return df


def verify_position_feature(frame_num: int, df: pd.DataFrame, video_paths: dict) -> None:
    """Non-interactive single-frame debug plot (matplotlib). Not part of the
    automated pipeline -- call manually if you want to eyeball one frame."""
    import matplotlib.pyplot as plt

    if frame_num not in df["Frame"].values:
        return
    row = df[df["Frame"] == frame_num].iloc[0]

    position = row.get("Doctor_Position", "Unknown")
    vid = video_paths.get("Side")
    if not vid:
        return

    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return

    cv2.rectangle(frame, (10, 10), (450, 70), (0, 0, 0), -1)
    cv2.putText(frame, f"Doc Position: {position}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    doc_hx, doc_hy = row.get("Side_Doctor_L_Hip_X"), row.get("Side_Doctor_R_Hip_X")
    pat_sh_x = np.nanmean([row.get("Side_Patient_L_Shoulder_X"), row.get("Side_Patient_R_Shoulder_X")])
    pat_knee_x = np.nanmean([row.get("Side_Patient_L_Knee_X"), row.get("Side_Patient_R_Knee_X")])

    h, w, _ = frame.shape
    if not np.isnan(pat_sh_x):
        cv2.line(frame, (int(pat_sh_x), 0), (int(pat_sh_x), h), (255, 0, 0), 2)
        cv2.putText(frame, "Shoulder Line", (int(pat_sh_x) + 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if not np.isnan(pat_knee_x):
        cv2.line(frame, (int(pat_knee_x), 0), (int(pat_knee_x), h), (0, 0, 255), 2)
        cv2.putText(frame, "Knee Line", (int(pat_knee_x) + 10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    doc_x = np.nanmean([doc_hx, doc_hy])
    if not np.isnan(doc_x):
        cv2.circle(frame, (int(doc_x), int(h / 2)), 15, (0, 255, 255), -1)
        cv2.putText(frame, "Doc CM", (int(doc_x) - 30, int(h / 2) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 6))
    plt.imshow(frame_rgb)
    plt.title(f"Side View Spatial Tracking - Frame {frame_num}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. Audio Umbrella: Gemini transcript classification merged in
# ==========================================
def time_to_frames(time_str: str, fps: int = 24) -> int:
    time_str = time_str.strip("[]")
    m, s = time_str.split(":")
    total_seconds = int(m) * 60 + int(s)
    return int(total_seconds * fps)


def classify_transcript_with_llm(transcript_path: str, gemini_api_key: str,
                                  model_name: str = "gemini-2.5-flash") -> list:
    """NOTE: source notebook hardcoded model_name="gemini-3-flash-preview", which
    404'd (idx 100); the working model used elsewhere in the same notebook was
    "gemini-2.5-flash" (idx 101) -- used as the default here instead."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gemini_api_key)

    with open(transcript_path, "r") as file:
        raw_transcript = file.read()

    prompt = f"""
    You are a medical exam classifier. Read the following transcript of a cardiovascular examination.
    Map each timestamped sentence to one of these 8 Umbrella Categories:
    1. Introduction and Preparation
    2. General Inspection
    3. Pulses and Neck Examination
    4. Precordium Palpation
    5. Supine Auscultation
    6. Left Lateral Maneuvers
    7. Sitting and Back Examination
    8. Peripheral Check and Conclusion

    Return ONLY a valid JSON array of objects with the exact keys: "start_time", "end_time", "text", and "umbrella_category".

    Transcript:
    {raw_transcript}
    """

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    import json
    return json.loads(response.text)


def merge_llm_features(csv_path: str, transcript_path: str, output_path: str,
                        gemini_api_key: str, fps: int = 24) -> None:
    print("Calling API for semantic classification...")
    llm_data = classify_transcript_with_llm(transcript_path, gemini_api_key)

    df = pd.read_csv(csv_path)
    df["Audio_Umbrella_Prediction"] = "None"

    print("Mapping LLM predictions to frames...")
    for entry in llm_data:
        start_frame = time_to_frames(entry["start_time"], fps)
        end_frame = time_to_frames(entry["end_time"], fps)
        mask = (df["Frame"] >= start_frame) & (df["Frame"] <= end_frame)
        df.loc[mask, "Audio_Umbrella_Prediction"] = entry["umbrella_category"]

    df["Audio_Umbrella_Prediction"] = df["Audio_Umbrella_Prediction"].replace("None", pd.NA).ffill().fillna("None")

    df.to_csv(output_path, index=False)
    print(f"Saved enriched dataset to {output_path}")


# ==========================================
# 5. Hand velocity + anatomical zone distance features
# ==========================================
def add_velocity_features(file_path: str) -> None:
    df = pd.read_csv(file_path)
    views = ["Front", "Side"]

    for view in views:
        dx_l = df[f"{view}_Doctor_L_Wrist_X"].diff()
        dy_l = df[f"{view}_Doctor_L_Wrist_Y"].diff()
        df[f"{view}_L_Vel"] = np.sqrt(dx_l ** 2 + dy_l ** 2).fillna(0)

        dx_r = df[f"{view}_Doctor_R_Wrist_X"].diff()
        dy_r = df[f"{view}_Doctor_R_Wrist_Y"].diff()
        df[f"{view}_R_Vel"] = np.sqrt(dx_r ** 2 + dy_r ** 2).fillna(0)

        df[f"{view}_L_Smooth"] = df[f"{view}_L_Vel"].rolling(window=15, center=True).mean().fillna(0)
        df[f"{view}_R_Smooth"] = df[f"{view}_R_Vel"].rolling(window=15, center=True).mean().fillna(0)

    df["L_Wrist_Max_Velocity"] = df[["Front_L_Smooth", "Side_L_Smooth"]].max(axis=1)
    df["R_Wrist_Max_Velocity"] = df[["Front_R_Smooth", "Side_R_Smooth"]].max(axis=1)
    df["Doctor_Overall_Hand_Movement"] = df[["L_Wrist_Max_Velocity", "R_Wrist_Max_Velocity"]].max(axis=1)

    cols_to_drop = [f"{v}_{side}_{t}" for v in views for side in ["L", "R"] for t in ["Vel", "Smooth"]]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(file_path, index=False)
    print(f"Hand velocity features added and saved to {file_path}")


def add_anatomical_zones(file_path: str) -> None:
    df = pd.read_csv(file_path)
    views = ["Front", "Side"]

    for view in views:
        df[f"{view}_Pat_Chest_X"] = (df[f"{view}_Patient_L_Shoulder_X"] + df[f"{view}_Patient_R_Shoulder_X"]) / 2
        df[f"{view}_Pat_Chest_Y"] = (df[f"{view}_Patient_L_Shoulder_Y"] + df[f"{view}_Patient_R_Shoulder_Y"]) / 2
        df[f"{view}_Pat_Neck_X"] = df[f"{view}_Patient_Nose_X"]
        df[f"{view}_Pat_Neck_Y"] = df[f"{view}_Patient_Nose_Y"]

        for doc_wrist in ["L", "R"]:
            wx = df[f"{view}_Doctor_{doc_wrist}_Wrist_X"]
            wy = df[f"{view}_Doctor_{doc_wrist}_Wrist_Y"]

            df[f"{view}_Dist_W{doc_wrist}_Chest"] = np.sqrt(
                (wx - df[f"{view}_Pat_Chest_X"]) ** 2 + (wy - df[f"{view}_Pat_Chest_Y"]) ** 2)
            df[f"{view}_Dist_W{doc_wrist}_Neck"] = np.sqrt(
                (wx - df[f"{view}_Pat_Neck_X"]) ** 2 + (wy - df[f"{view}_Pat_Neck_Y"]) ** 2)

            dist_to_l_wrist = np.sqrt(
                (wx - df[f"{view}_Patient_L_Wrist_X"]) ** 2 + (wy - df[f"{view}_Patient_L_Wrist_Y"]) ** 2)
            dist_to_r_wrist = np.sqrt(
                (wx - df[f"{view}_Patient_R_Wrist_X"]) ** 2 + (wy - df[f"{view}_Patient_R_Wrist_Y"]) ** 2)
            df[f"{view}_Dist_W{doc_wrist}_PatWrists"] = np.minimum(dist_to_l_wrist, dist_to_r_wrist)

    df["Min_Dist_to_Chest"] = df[["Front_Dist_WL_Chest", "Front_Dist_WR_Chest",
                                    "Side_Dist_WL_Chest", "Side_Dist_WR_Chest"]].min(axis=1)
    df["Min_Dist_to_Neck"] = df[["Front_Dist_WL_Neck", "Front_Dist_WR_Neck",
                                   "Side_Dist_WL_Neck", "Side_Dist_WR_Neck"]].min(axis=1)
    df["Min_Dist_to_PatWrists"] = df[["Front_Dist_WL_PatWrists", "Front_Dist_WR_PatWrists",
                                        "Side_Dist_WL_PatWrists", "Side_Dist_WR_PatWrists"]].min(axis=1)

    cols_to_drop = [c for c in df.columns if "Dist_W" in c or "_Pat_Chest_" in c or "_Pat_Neck_" in c]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(file_path, index=False)
    print(f"Anatomical zones added and saved to {file_path}")


# ==========================================
# 6. Continuous geometric zone projection (idx 130 -- latest version, supersedes idx 129)
# ==========================================
def get_continuous_side_vertical(wx, wy, N, S, H, torso_length):
    if pd.isna(wx) or pd.isna(wy):
        return None, "N/A"
    W = np.array([wx, wy])

    v_neck = S - N
    if np.dot(v_neck, v_neck) > 0:
        t_neck = np.dot(W - N, v_neck) / np.dot(v_neck, v_neck)
        proj_neck = N + t_neck * v_neck
        if 0 <= t_neck <= 1.0 and np.linalg.norm(W - proj_neck) < (0.35 * torso_length):
            return float(t_neck - 1.0), "Neck"

    v_torso = H - S
    len_torso_sq = np.dot(v_torso, v_torso)
    if len_torso_sq > 0:
        t_torso = np.dot(W - S, v_torso) / len_torso_sq
        proj_torso = S + t_torso * v_torso

        if -0.1 <= t_torso <= 1.1 and np.linalg.norm(W - proj_torso) < (0.85 * torso_length):
            if t_torso <= 0.35:
                cat = "Upper_Chest"
            elif t_torso <= 0.60:
                cat = "Lower_Chest"
            elif t_torso <= 0.85:
                cat = "Stomach"
            else:
                cat = "Pelvis"
            return float(t_torso), cat

    return None, "N/A"


def get_continuous_front_horizontal(wx, wy, S, H, shoulder_width):
    if pd.isna(wx) or pd.isna(wy) or shoulder_width == 0:
        return None, "N/A"
    W = np.array([wx, wy])

    v_torso = H - S
    len_torso_sq = np.dot(v_torso, v_torso)
    if len_torso_sq > 0:
        t_torso = np.dot(W - S, v_torso) / len_torso_sq
        proj_torso = S + t_torso * v_torso

        lateral_dist = W[0] - proj_torso[0]
        norm_lat = lateral_dist / (shoulder_width / 2)

        if abs(norm_lat) <= 0.20:
            return float(norm_lat), "Center"
        elif norm_lat < 0:
            return float(norm_lat), "Left"
        else:
            return float(norm_lat), "Right"

    return None, "N/A"


def process_frame_row(row) -> pd.Series:
    features = {
        "L_Side_T": np.nan, "L_Side_Zone": "N/A",
        "R_Side_T": np.nan, "R_Side_Zone": "N/A",
        "L_Front_X": np.nan, "L_Front_Zone": "N/A",
        "R_Front_X": np.nan, "R_Front_Zone": "N/A",
        "Min_Pulse_Dist": 999.0,
    }

    for view in ["Front", "Side"]:
        lsx, lsy = row.get(f"{view}_Patient_L_Shoulder_X"), row.get(f"{view}_Patient_L_Shoulder_Y")
        rsx, rsy = row.get(f"{view}_Patient_R_Shoulder_X"), row.get(f"{view}_Patient_R_Shoulder_Y")
        lhx, lhy = row.get(f"{view}_Patient_L_Hip_X"), row.get(f"{view}_Patient_L_Hip_Y")
        rhx, rhy = row.get(f"{view}_Patient_R_Hip_X"), row.get(f"{view}_Patient_R_Hip_Y")
        nx, ny = row.get(f"{view}_Patient_Nose_X"), row.get(f"{view}_Patient_Nose_Y")

        pw_lx, pw_ly = row.get(f"{view}_Patient_L_Wrist_X"), row.get(f"{view}_Patient_L_Wrist_Y")
        pw_rx, pw_ry = row.get(f"{view}_Patient_R_Wrist_X"), row.get(f"{view}_Patient_R_Wrist_Y")

        if pd.notna([lsx, lsy, rsx, rsy, lhx, lhy, rhx, rhy, nx, ny]).all():
            S = np.array([(lsx + rsx) / 2, (lsy + rsy) / 2])
            H = np.array([(lhx + rhx) / 2, (lhy + rhy) / 2])
            N = np.array([nx, ny])
            v_torso = H - S
            torso_length = np.linalg.norm(v_torso)
            shoulder_width = np.linalg.norm(np.array([lsx, lsy]) - np.array([rsx, rsy]))

            for hand in ["L", "R"]:
                wx, wy = row.get(f"{view}_Doctor_{hand}_Wrist_X"), row.get(f"{view}_Doctor_{hand}_Wrist_Y")
                ex, ey = row.get(f"{view}_Doctor_{hand}_Elbow_X"), row.get(f"{view}_Doctor_{hand}_Elbow_Y")

                if pd.notna([wx, wy, ex, ey]).all():
                    palm_x = wx + (wx - ex) * 0.30
                    palm_y = wy + (wy - ey) * 0.30
                    palm_arr = np.array([palm_x, palm_y])

                    for px, py in [(pw_lx, pw_ly), (pw_rx, pw_ry)]:
                        if pd.notna([px, py]).all():
                            d = np.linalg.norm(palm_arr - np.array([px, py]))
                            if d < features["Min_Pulse_Dist"]:
                                features["Min_Pulse_Dist"] = d

                    if view == "Side":
                        val_y, cat_y = get_continuous_side_vertical(palm_x, palm_y, N, S, H, torso_length)
                        if val_y is not None:
                            features["L_Side_T" if hand == "L" else "R_Side_T"] = val_y
                            features["L_Side_Zone" if hand == "L" else "R_Side_Zone"] = cat_y

                    elif view == "Front":
                        val_x, cat_x = get_continuous_front_horizontal(palm_x, palm_y, S, H, shoulder_width)
                        if val_x is not None:
                            features["L_Front_X" if hand == "L" else "R_Front_X"] = val_x
                            features["L_Front_Zone" if hand == "L" else "R_Front_Zone"] = cat_x

    def combine(f_cat, s_cat):
        if f_cat == "N/A" and s_cat == "N/A":
            return "N/A"
        if f_cat == "N/A":
            return s_cat
        if s_cat == "N/A":
            return f_cat
        return f"{f_cat}_{s_cat}"

    features["L_Combined_Zone"] = combine(features["L_Front_Zone"], features["L_Side_Zone"])
    features["R_Combined_Zone"] = combine(features["R_Front_Zone"], features["R_Side_Zone"])

    return pd.Series(features)


def update_master_dataset(csv_path: str) -> None:
    print("Reading dataset and computing continuous geometric features...")
    df = pd.read_csv(csv_path)

    new_features_df = df.apply(process_frame_row, axis=1)
    for col in new_features_df.columns:
        df[col] = new_features_df[col]

    df["Min_Pulse_Dist"] = df["Min_Pulse_Dist"].replace(999.0, np.nan)

    df.to_csv(csv_path, index=False)
    print(f"Master dataset updated successfully at: {csv_path}")
