"""
YOLOv8 batched pose extraction.

Ported from FS_model.ipynb (Section 5: YOLOv8 Pose Extraction, cell idx 17).
Confirmed working: real batched run on Take 5's 360 view (idx 27), producing
360_Keypoints.csv, later scanned successfully by the ID tracker (idx 29).
"""
import os
import shutil

import cv2
import pandas as pd
import torch
from tqdm.auto import tqdm
from ultralytics import YOLO

KEYPOINT_NAMES = [
    "Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "L_Hip", "R_Hip",
    "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
]


def get_best_device():
    if torch.cuda.is_available():
        return 0
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_pose_extraction_batched(video_paths, save_paths_video, save_paths_csv,
                                 model_name: str = "yolov8x-pose.pt", batch_size: int = 50,
                                 frame_skip: int = 1) -> None:
    """Batched YOLOv8 pose+track inference, writing one keypoints CSV per input video.

    `save_paths_video` is accepted for interface compatibility with the
    original notebook but is currently unused (no skeleton video is written
    here -- see skeleton_viz.py for that).
    """
    device = get_best_device()
    print(f"Acceleration device: {device} (batch size: {batch_size})")

    print(f"Loading model: {model_name}...")
    model = YOLO(model_name)

    for video_path, save_path_video, save_path_csv in zip(video_paths, save_paths_video, save_paths_csv):
        local_video_input = "/content/temp_processing_input.mp4"
        local_csv_output = "/content/temp_processing_output.csv"

        filename = os.path.basename(video_path)
        print("\n------------------------------------------------")
        print(f"Source: {filename}")

        if os.path.exists(video_path):
            print("Copying to local disk for speed (approx 30s)...")
            shutil.copy(video_path, local_video_input)
        else:
            print(f"Error: file not found at {video_path}")
            continue

        cap = cv2.VideoCapture(local_video_input)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_keypoints = []
        frame_idx = 0
        batch_frames = []

        print(f"Starting batch inference on {total_frames} frames...")

        with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                batch_frames.append(frame)

                if len(batch_frames) == batch_size or frame_idx + len(batch_frames) == total_frames:
                    results = model.track(batch_frames, persist=True, verbose=False, device=device)

                    for i, r in enumerate(results):
                        current_frame_num = frame_idx + i

                        if current_frame_num % frame_skip == 0:
                            if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                                kpts = r.keypoints.xy.cpu().numpy()
                                track_ids = r.boxes.id.cpu().numpy().astype(int)

                                for j in range(len(track_ids)):
                                    row = [current_frame_num, track_ids[j]] + kpts[j].flatten().tolist()
                                    all_keypoints.append(row)

                    processed_count = len(batch_frames)
                    frame_idx += processed_count
                    pbar.update(processed_count)

                    batch_frames = []

        cap.release()

        print("Formatting CSV data...")
        cols = ["Frame", "Person_ID"]
        for name in KEYPOINT_NAMES:
            cols.extend([f"{name}_X", f"{name}_Y"])

        df = pd.DataFrame(all_keypoints, columns=cols)
        df.to_csv(local_csv_output, index=False)

        print(f"Moving results back to Drive: {os.path.basename(save_path_csv)}")
        shutil.move(local_csv_output, save_path_csv)

        if os.path.exists(local_video_input):
            os.remove(local_video_input)

        print("Finished processing.")

    print("\nAll videos processed!")
