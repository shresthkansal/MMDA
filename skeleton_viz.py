"""
Non-interactive skeleton drawing and preview-video rendering.

Ported from FS_model.ipynb (Section 7: Skeleton Visualization, cell idx 21;
and the preview generator duplicated identically in idx 31 and idx 33 --
idx 33's run is the one with confirmed successful output,
"Rendering: 8760/8909 (98.3%) ... Preview saved").

Two independent skeleton-drawing implementations exist for two different
consumers and are both kept: `draw_skeletons_on_frame` (role-colored,
index-pair based, used by `create_labeled_video`) and
`draw_enhanced_skeletons` (name-pair based, doctor/patient-colored, used by
`create_preview_video`).
"""
import os

import cv2
import pandas as pd
from google.colab.patches import cv2_imshow
from IPython.display import clear_output

COL_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]
KPT_X_COLS = [f"{n}_X" for n in COL_NAMES]
KPT_Y_COLS = [f"{n}_Y" for n in COL_NAMES]

SKELETON_PAIRS = [
    [0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [11, 12], [5, 11], [6, 12], [11, 13], [13, 15], [12, 14], [14, 16],
]

LABEL_COLORS = {
    "doctor": (255, 100, 0),
    "patient": (0, 255, 0),
    "default": (200, 200, 200),
}


# ==========================================
# Variant 1: role-colored, used by create_labeled_video
# ==========================================
def draw_skeletons_on_frame(frame, frame_data):
    for _, person in frame_data.iterrows():
        raw_label = person.get("Role", "default")
        label = "default" if pd.isna(raw_label) else str(raw_label)
        color = LABEL_COLORS.get(label.lower(), LABEL_COLORS["default"])

        xs = person[KPT_X_COLS].values.astype(int)
        ys = person[KPT_Y_COLS].values.astype(int)
        kpts = list(zip(xs, ys))

        for i, j in SKELETON_PAIRS:
            if kpts[i][0] > 0 and kpts[j][0] > 0:
                cv2.line(frame, kpts[i], kpts[j], color, 2, cv2.LINE_AA)
        for x, y in kpts:
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 4, color, -1, cv2.LINE_AA)

        label_pos = kpts[0] if kpts[0][0] > 0 else kpts[5]
        if label_pos[0] > 0:
            cv2.putText(frame, label, (label_pos[0], label_pos[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def create_labeled_video(input_video_path: str, input_csv_path: str, output_video_path: str,
                          out_fps: float = None) -> None:
    print(f"Processing {os.path.basename(input_video_path)}...")

    try:
        df = pd.read_csv(input_csv_path)
        df = df.sort_values(["Person_ID", "Frame"])

        df_list = []
        for pid in df["Person_ID"].unique():
            d_pid = df[df["Person_ID"] == pid].set_index("Frame")
            full_range = range(d_pid.index.min(), d_pid.index.max() + 1)
            d_pid = d_pid.reindex(full_range)
            d_pid["Person_ID"] = pid
            d_pid["Role"] = d_pid["Role"].ffill().bfill().fillna("Doctor")
            d_pid[KPT_X_COLS + KPT_Y_COLS] = d_pid[KPT_X_COLS + KPT_Y_COLS].ffill(limit=15)
            df_list.append(d_pid.reset_index().rename(columns={"index": "Frame"}))

        if not df_list:
            print("No valid data found in CSV.")
            return

        df_smoothed = pd.concat(df_list).dropna(subset=["Nose_X"])
        frame_groups = df_smoothed.groupby("Frame")

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_fps = out_fps or fps
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"MJPG"), out_fps, (w, h))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in frame_groups.groups:
            frame_data = frame_groups.get_group(frame_idx)
            frame = draw_skeletons_on_frame(frame, frame_data)

        out.write(frame)
        if frame_idx % 200 == 0:
            print(f"  ... {frame_idx}/{total_vid_frames}", end="\r")
        frame_idx += 1

    cap.release()
    out.release()
    print(f"\nVideo saved: {os.path.basename(output_video_path)}")


# ==========================================
# Variant 2: doctor/patient-colored, used by create_preview_video
# ==========================================
def draw_enhanced_skeletons(frame, frame_data):
    connections = [
        ("Nose", "L_Shoulder"), ("Nose", "R_Shoulder"), ("L_Shoulder", "R_Shoulder"),
        ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"), ("R_Shoulder", "R_Elbow"),
        ("R_Elbow", "R_Wrist"), ("L_Shoulder", "L_Hip"), ("R_Shoulder", "R_Hip"),
        ("L_Hip", "R_Hip"), ("L_Hip", "L_Knee"), ("R_Hip", "R_Knee"),
        ("L_Knee", "L_Ankle"), ("R_Knee", "R_Ankle"),
    ]
    for _, person in frame_data.iterrows():
        role = person.get("Role", "Unknown")
        if role == "Noise":
            continue
        color = (255, 100, 0) if role == "Doctor" else (0, 255, 100)
        for s, e in connections:
            try:
                p1 = (int(person[f"{s}_X"]), int(person[f"{s}_Y"]))
                p2 = (int(person[f"{e}_X"]), int(person[f"{e}_Y"]))
                if p1[0] > 0 and p2[0] > 0:
                    cv2.line(frame, p1, p2, color, 3)
            except Exception:
                continue
        try:
            cv2.putText(frame, f"{role}", (int(person["Nose_X"]), int(person["Nose_Y"] - 30)),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, color, 2)
        except Exception:
            continue
    return frame


def create_preview_video(input_video_path, input_csv_path, output_video_path,
                          frame_range=2000, draw_skeleton_fn=None,
                          scale=0.3, display_in_colab=True, display_freq=50, display_scale=0.5) -> None:
    """Interpolates based on Canonical_ID (Role) to prevent ghosting during ID switches."""
    print(f"[DEBUG] Loading CSV: {input_csv_path}...")
    df = pd.read_csv(input_csv_path)

    if frame_range is None or frame_range == "all":
        start_frame, end_frame = int(df["Frame"].min()), int(df["Frame"].max())
    elif isinstance(frame_range, tuple):
        start_frame, end_frame = int(frame_range[0]), int(frame_range[1])
    else:
        start_frame = int(df["Frame"].min())
        end_frame = start_frame + int(frame_range)

    print(f"[DEBUG] Target Range: {start_frame} to {end_frame}")

    buffer = 50
    subset_mask = (df["Frame"] >= start_frame - buffer) & (df["Frame"] <= end_frame + buffer)
    df_subset = df[subset_mask].copy()

    print("[DEBUG] Smoothing data by ROLE (Canonical ID)...")
    smoothed_parts = []
    for cid in [1, 2]:
        role_data = df_subset[df_subset["Canonical_ID"] == cid].sort_values("Frame").set_index("Frame")
        if role_data.empty:
            continue
        local_min, local_max = int(role_data.index.min()), int(role_data.index.max())
        full_idx = range(local_min, local_max + 1)
        role_data = role_data.reindex(full_idx)
        role_data["Canonical_ID"] = cid
        role_data["Role"] = "Doctor" if cid == 1 else "Patient"
        kpt_cols = [c for c in role_data.columns if "_X" in c or "_Y" in c]
        role_data[kpt_cols] = role_data[kpt_cols].interpolate(method="linear", limit=10)
        smoothed_parts.append(role_data.reset_index().rename(columns={"index": "Frame"}))

    if smoothed_parts:
        df_smoothed = pd.concat(smoothed_parts).dropna(subset=["Nose_X"])
        frame_groups = df_smoothed.groupby("Frame")
    else:
        frame_groups = df_subset.groupby("Frame")

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))

    current_frame = 0
    while current_frame < start_frame:
        ret = cap.grab()
        if not ret:
            break
        current_frame += 1
        if current_frame % 2000 == 0:
            print(f"    -> Skipping... {current_frame}", end="\r")

    print(f"\n[DEBUG]    -> Starting Render at {current_frame}...")

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in frame_groups.groups:
            frame_data = frame_groups.get_group(current_frame)
            if draw_skeleton_fn:
                frame = draw_skeleton_fn(frame, frame_data)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out.write(resized)

        if current_frame % display_freq == 0:
            prog = (current_frame - start_frame) / (end_frame - start_frame) * 100
            status = f"Rendering: {current_frame}/{end_frame} ({prog:.1f}%)"
            if display_in_colab:
                clear_output(wait=True)
                print(status)
                vis_frame = cv2.resize(frame, (int(orig_w * display_scale), int(orig_h * display_scale)))
                cv2_imshow(vis_frame)
            else:
                print(status, end="\r")
        current_frame += 1

    cap.release()
    out.release()
    print(f"\nPreview saved: {output_video_path}")
