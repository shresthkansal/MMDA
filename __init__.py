"""
osce_pipeline: consolidated, deduplicated port of the project's research
notebooks (FS_model.ipynb, and in-progress Data_Preparation.ipynb) into a
clean Python package.

Modules:
    config              -- take/environment configuration and path building
    sync                -- audio sync offset calculation
    render              -- ffmpeg video stitching/rendering
    pose                -- YOLOv8 batched pose extraction
    reid                -- person re-identification / doctor-patient role assignment
    skeleton_viz        -- non-interactive skeleton drawing / preview rendering
    action_library      -- transcript + geometric zone-prediction "action library" building
                           (flagged as likely superseded by reference_library's Qwen-VLM approach)
    grading             -- FS_model's own grading prototype (overlaps Data_Prep's Qwen approach)
    features            -- the confirmed-good final feature-engineering chain
    diagnostics         -- ReID tracking QA utilities
    phase0              -- Data_Prep Phase 0: taxonomy/anchor validation + anchor embeddings
    reference_library   -- Data_Prep Phase 1: Qwen-VLM reference action-library builder
    run                 -- thin per-take orchestrator (FS_model chain only; phase0/
                           reference_library run standalone against the reference take)
"""
