"""
QA / diagnostic utilities for checking ReID tracker output quality.

Ported from FS_model.ipynb (ad hoc analysis cells idx 32, 36, 37, 38, all
confirmed to run successfully against real Take 2/3/5/6 data). These are
not part of the automated pipeline -- run manually after reid.py to sanity
check tracking quality before proceeding to feature engineering.

The batch-loop scripts (originally relying on notebook globals like
TARGET_TAKES/P/VIEW set in earlier cells) have been wrapped into proper
functions taking take_ids/views as parameters.
"""
import os

import numpy as np
import pandas as pd


def analyze_interaction_quality(csv_path: str, take_idx, view_name: str, plot: bool = True) -> None:
    """Prints doctor/patient co-presence stats and average inter-person distance
    for a labeled ReID CSV; optionally plots a tracking-continuity heatmap."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    frame_status = df.groupby(["Frame", "Role"]).size().unstack(fill_value=0)
    if "Doctor" not in frame_status.columns:
        frame_status["Doctor"] = 0
    if "Patient" not in frame_status.columns:
        frame_status["Patient"] = 0

    frame_status["Has_Doc"] = frame_status["Doctor"] > 0
    frame_status["Has_Pat"] = frame_status["Patient"] > 0
    frame_status["Both_Present"] = frame_status["Has_Doc"] & frame_status["Has_Pat"]

    min_frame = int(df["Frame"].min())
    max_frame = int(df["Frame"].max())
    total_frames = max_frame - min_frame

    coexist_count = frame_status["Both_Present"].sum()
    doc_only = (frame_status["Has_Doc"] & ~frame_status["Has_Pat"]).sum()
    pat_only = (~frame_status["Has_Doc"] & frame_status["Has_Pat"]).sum()

    print(f"\nREPORT: Take {take_idx} - {view_name}")
    print("=" * 40)
    print(f"  Duration:      {total_frames} frames")
    print(f"  Both visible:  {coexist_count} frames ({coexist_count / total_frames:.1%})")

    if doc_only > 50:
        print(f"  Doctor only:  {doc_only} frames (patient missing)")
    if pat_only > 50:
        print(f"  Patient only: {pat_only} frames (doctor missing)")

    interaction_frames = df[df["Frame"].isin(frame_status[frame_status["Both_Present"]].index)]
    if not interaction_frames.empty:
        doc_pos = interaction_frames[interaction_frames["Role"] == "Doctor"][["Frame", "Nose_X", "Nose_Y"]].set_index("Frame")
        pat_pos = interaction_frames[interaction_frames["Role"] == "Patient"][["Frame", "Nose_X", "Nose_Y"]].set_index("Frame")

        merged = doc_pos.join(pat_pos, lsuffix="_doc", rsuffix="_pat")
        merged["Distance"] = np.sqrt(
            (merged["Nose_X_doc"] - merged["Nose_X_pat"]) ** 2 +
            (merged["Nose_Y_doc"] - merged["Nose_Y_pat"]) ** 2
        )
        avg_dist = merged["Distance"].mean()
        print(f"  Avg. dist:    {avg_dist:.1f} px")
    else:
        print("  Avg. dist:    N/A (no co-existence)")

    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(12, 3))
        full_timeline = frame_status[["Has_Doc", "Has_Pat"]].reindex(
            range(min_frame, max_frame + 1), fill_value=False
        ).T.astype(int)

        sns.heatmap(full_timeline, cmap=["#ffebee", "#4caf50"], cbar=False, yticklabels=["Doctor", "Patient"])
        plt.title(f"Take {take_idx} ({view_name}): Tracking Continuity", fontsize=10)
        plt.xlabel("Frame")
        plt.yticks(rotation=0)
        plt.show()


def run_batch_interaction_analysis(get_take_paths_fn, take_ids: list, views: list = ("side", "front", "360"),
                                    plot: bool = True) -> None:
    print(f"Starting batch analysis for takes: {take_ids}\n")
    for take in take_ids:
        p = get_take_paths_fn(take)
        for view in views:
            csv_key = f"pose_{view}_labeled_csv"
            if csv_key not in p:
                continue
            csv_path = p[csv_key]
            if os.path.exists(csv_path):
                analyze_interaction_quality(csv_path, take, view, plot=plot)
            else:
                print(f"\nFile not found: Take {take} - {view}")


def check_overlapping_roles(csv_path: str) -> None:
    """Confirms at most one Doctor and one Patient per frame."""
    df_debug = pd.read_csv(csv_path)
    print(f"Max frame: {df_debug['Frame'].max()}")

    doc_counts = df_debug[df_debug["Role"] == "Doctor"].groupby("Frame").size()
    pat_counts = df_debug[df_debug["Role"] == "Patient"].groupby("Frame").size()

    double_doc = doc_counts[doc_counts > 1]
    double_pat = pat_counts[pat_counts > 1]

    print("=" * 40)
    print("DUPLICATE ROLE DIAGNOSIS")
    print("=" * 40)

    if len(double_doc) > 0:
        print(f"FOUND {len(double_doc)} frames with multiple doctors! Examples: {double_doc.head().index.tolist()}")
    else:
        print("Doctor role is clean (max 1 per frame).")

    if len(double_pat) > 0:
        print(f"FOUND {len(double_pat)} frames with multiple patients! Examples: {double_pat.head().index.tolist()}")
    else:
        print("Patient role is clean (max 1 per frame).")


def check_end_of_video_gaps(get_take_paths_fn, take_ids: list, views: list = ("360", "front", "side")) -> None:
    """Reports the frame gap between doctor and patient tracking's last-seen frame,
    per take/view -- large gaps suggest one role's track ended early."""
    print(f"{'Take':<6} | {'View':<6} | {'Last Doc':<10} | {'Last Pat':<10} | {'Gap (Frames)':<12} | {'Status'}")
    print("-" * 75)

    for take in take_ids:
        p = get_take_paths_fn(take)
        for view in views:
            csv_key = f"pose_{view}_labeled_csv"
            if csv_key not in p:
                print(f"{take:<6} | {view:<6} | {'N/A':<10} | {'N/A':<10} | {'-':<12} | Path key missing")
                continue

            csv_path = p[csv_key]
            if not os.path.exists(csv_path):
                print(f"{take:<6} | {view:<6} | {'N/A':<10} | {'N/A':<10} | {'-':<12} | File not found")
                continue

            try:
                df = pd.read_csv(csv_path)
                doc_frames = df[df["Role"] == "Doctor"]["Frame"]
                pat_frames = df[df["Role"] == "Patient"]["Frame"]

                last_doc = doc_frames.max() if not doc_frames.empty else 0
                last_pat = pat_frames.max() if not pat_frames.empty else 0

                if last_doc == 0 or last_pat == 0:
                    gap, status = "N/A", "Missing role"
                else:
                    gap = abs(last_doc - last_pat)
                    status = "Perfect" if gap == 0 else ("Minor gap" if gap < 50 else "Large gap")

                print(f"{take:<6} | {view:<6} | {int(last_doc):<10} | {int(last_pat):<10} | {gap:<12} | {status}")
            except Exception:
                print(f"{take:<6} | {view:<6} | {'Error':<10} | {'Error':<10} | {'-':<12} | Read error")


def get_missing_ranges(csv_path: str) -> "dict | None":
    """Returns {'Doctor': [...], 'Patient': [...]} of (start,end) frame ranges > 50
    frames long where that role was not tracked."""
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    frame_status = df.groupby(["Frame", "Role"]).size().unstack(fill_value=0)
    if "Doctor" not in frame_status.columns:
        frame_status["Doctor"] = 0
    if "Patient" not in frame_status.columns:
        frame_status["Patient"] = 0

    frame_status["Has_Doc"] = frame_status["Doctor"] > 0
    frame_status["Has_Pat"] = frame_status["Patient"] > 0

    min_frame = int(df["Frame"].min())
    report = {"Doctor": [], "Patient": []}

    for role, label in [("Has_Doc", "Doctor"), ("Has_Pat", "Patient")]:
        mask = frame_status[role].astype(int)
        diff = np.diff(np.concatenate(([1], mask, [1])))
        starts = np.where(diff == -1)[0] + min_frame
        ends = np.where(diff == 1)[0] + min_frame - 1

        for s, e in zip(starts, ends):
            duration = e - s
            if duration > 50:
                report[label].append(f"({s},{e}) ({duration} frms)")

    return report


def check_missing_frame_ranges(get_take_paths_fn, take_ids: list, views: list = ("360", "front", "side")) -> None:
    print(f"{'TAKE':<6} | {'VIEW':<8} | {'ROLE':<10} | {'MISSING FRAME RANGES TO CHECK'}")
    print("=" * 85)

    for take in take_ids:
        p = get_take_paths_fn(take)
        for view in views:
            csv_key = f"pose_{view}_labeled_csv"
            if csv_key not in p:
                continue

            csv_path = p[csv_key]
            if not os.path.exists(csv_path):
                print(f"{take:<6} | {view:<8} | {'-':<10} | File not found")
                continue

            ranges = get_missing_ranges(csv_path)
            if ranges:
                if ranges["Doctor"]:
                    print(f"{take:<6} | {view:<8} | {'Doctor':<10} | {', '.join(ranges['Doctor'])}")
                if ranges["Patient"]:
                    print(f"{take:<6} | {view:<8} | {'Patient':<10} | {', '.join(ranges['Patient'])}")
                if not ranges["Doctor"] and not ranges["Patient"]:
                    print(f"{take:<6} | {view:<8} | {'Both':<10} | Perfect tracking (no major gaps)")

        print("-" * 85)
