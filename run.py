"""
Thin orchestrator chaining the pipeline stages for one take.

IMPORTANT: the ReID stage (reid.py) is semi-supervised by design -- a human
must look at `auto_generate_known_phases()`'s output and confirm/edit it
before `build_anchor_schedule()`/`run_id_based_tracker()` can run (this is
how the original notebook's confirmed-working tracking chain operated; see
reid.py's module docstring). So there is no single fully-unattended
`run(take_id)` -- the pipeline is split into an automated front half, a
human-in-the-loop ReID step, and an automated back half (features).

action_library.py and grading.py are intentionally NOT auto-chained here:
their value is still under review (known-weak zone classification;
grading.py functionally overlaps with the Data_Prep Qwen approach). Call
them directly if/when needed.
"""
from . import config, features, pose, reid, render

# `sync.run_sync_calculator` is available for computing fresh offsets for a
# take not yet in config.SYNC_CONFIG, but isn't auto-chained here -- the
# confirmed-working notebook cells (idx 23, 25) consumed precomputed
# SYNC_CONFIG values via config.get_sync_config(), not a live calculator call.


def run_automated_front_half(take_id: int, view: str = "360",
                              pose_model: str = "yolov8l-pose.pt", batch_size: int = 64) -> dict:
    """Sync -> render -> pose extraction. Fully automated. Returns the take's
    path dict for convenience in the next (human-in-the-loop ReID) step."""
    config.mount_drive()
    p = config.get_take_paths(take_id)
    gpu_mode = config.get_environment_config() == "colab"

    sync_config = config.get_sync_config(take_id)
    files = config.get_take_files(take_id)
    raw_folder = p["raw_folder"]

    p_front = [config.resolve_media_path(raw_folder, f) for f in files["front"]]
    p_side = [config.resolve_media_path(raw_folder, f) for f in files["side"]]
    p_360 = [config.resolve_media_path(raw_folder, f, "") for f in files.get("360", [])]

    a_src_files = files[sync_config["audio_source_camera"]]
    aL_parts = [f"{raw_folder}/{f}{sync_config['audio_track_L']}.MXF" for f in a_src_files]
    aR_parts = [f"{raw_folder}/{f}{sync_config['audio_track_R']}.MXF" for f in a_src_files]

    side_clap = sync_config["start_trim_seconds"]
    ss_side = side_clap
    ss_front = max(side_clap - sync_config["offset_delay_seconds"], 0)
    ss_360 = max(side_clap - sync_config.get("360_offset_seconds", 0.0), 0)

    print(f"Rendering final camera outputs for Take {take_id}...")
    render.render_side_camera(p_side, aL_parts, aR_parts, p["vid_side"], ss_side, gpu_mode)
    render.render_front_camera(p_front, aL_parts, aR_parts, p["vid_front"], ss_front, ss_side, gpu_mode)
    if p_360:
        render.render_360_camera(p_360[0], aL_parts, aR_parts, p["vid_360"], ss_360, ss_side, gpu_mode)

    print(f"\nRunning pose extraction ({view} view)...")
    pose.run_pose_extraction_batched(
        video_paths=[p[f"vid_{view}"]],
        save_paths_video=["/content/dummy_skeleton.avi"],
        save_paths_csv=[p[f"pose_{view}_csv"]],
        model_name=pose_model,
        batch_size=batch_size,
    )

    return p


def reid_step_1_scan(take_id: int, view: str = "360") -> dict:
    """Human-in-the-loop ReID, step 1: scan for candidate ID-change phases.
    Review the returned dict (edit if needed) before passing to step 2."""
    p = config.get_take_paths(take_id)
    return reid.auto_generate_known_phases(p[f"pose_{view}_csv"])


def reid_step_2_and_3_track(take_id: int, known_phases: dict, view: str = "360") -> "object":
    """Human-in-the-loop ReID, steps 2+3: build the anchor schedule from the
    (human-reviewed) known_phases, then run the strict tracker and save the
    final labeled CSV."""
    p = config.get_take_paths(take_id)
    csv_path = p[f"pose_{view}_csv"]

    anchor_ids = reid.build_anchor_schedule(known_phases, csv_path)
    return reid.run_id_based_tracker(csv_path, p[f"pose_{view}_labeled_csv"], anchor_ids)


def run_feature_pipeline(take_id: int, gemini_api_key: str = None) -> None:
    """Features stage: assumes Front/Side/360 have all been through render +
    pose + the human-verified ReID step already (labeled CSVs exist).
    Runs the full confirmed-good chain from features.py in order.
    """
    p = config.get_take_paths(take_id)

    csv_paths = {
        "Front": p["pose_front_labeled_csv"],
        "Side": p["pose_side_labeled_csv"],
        "360": p["pose_360_labeled_csv"],
    }
    master_df = features.preprocess_medical_keypoints(csv_paths)
    master_df = features.add_smoothed_posture_features(master_df, window_size=15)
    master_df = features.add_strict_touch_features(master_df, threshold=75)
    master_df = features.add_doctor_position_feature(master_df)
    master_df.to_csv(p["wide_keypoints_csv"], index=False)

    if gemini_api_key:
        features.merge_llm_features(p["wide_keypoints_csv"], p["transcript"], p["wide_keypoints_csv"], gemini_api_key)
    else:
        print("Skipping merge_llm_features: no gemini_api_key provided.")

    features.add_velocity_features(p["wide_keypoints_csv"])
    features.add_anatomical_zones(p["wide_keypoints_csv"])
    features.update_master_dataset(p["wide_keypoints_csv"])

    print(f"\nFeature pipeline complete for Take {take_id}: {p['wide_keypoints_csv']}")
