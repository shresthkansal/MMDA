"""
osce_pipeline: consolidated, deduplicated port of FS_model.ipynb's unique
working code (Step 1 of the notebook -> package migration -- see
/Users/shresthkansal/.claude/plans/robust-rolling-allen.md for full context).

Modules:
    config          -- take/environment configuration and path building
    sync            -- audio sync offset calculation
    render          -- ffmpeg video stitching/rendering
    pose            -- YOLOv8 batched pose extraction
    reid            -- person re-identification / doctor-patient role assignment
    skeleton_viz    -- non-interactive skeleton drawing / preview rendering
    action_library  -- transcript + geometric zone-prediction "action library" building
    grading         -- FS_model's own grading prototype (overlaps Data_Prep's Qwen approach)
    features        -- the confirmed-good final feature-engineering chain
    diagnostics     -- ReID tracking QA utilities
    run             -- thin per-take orchestrator
"""
