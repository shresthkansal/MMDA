"""
Video stitching/rendering via ffmpeg (verification preview + per-camera final renders).

Ported from FS_model.ipynb (Section 4: Video Stitching & Rendering).

Two competing versions existed in the source notebook (cell idx 14 and idx 15).
Only idx 15's functions are ported here: real execution evidence (idx 23, 25)
shows `render_verification_preview`, `render_side_camera`, `render_front_camera`,
and `render_360_camera` are the functions actually invoked in every successful
render run. Cell 14's `render_verification_cpu/gpu` and
`render_stitched_camera_cpu/gpu` are never called anywhere in the notebook's
confirmed run history -- dead code, not ported.
"""
import platform
import subprocess
import sys


# ==========================================
# Utils: command runner & encoder selection
# ==========================================
def _run_cmd(cmd: str) -> None:
    """Streams ffmpeg output in real-time; raises on non-zero exit."""
    print("   Launching ffmpeg...")
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=0
    )
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip())
            sys.stdout.flush()
    if process.returncode != 0:
        print("Command failed.")
        raise subprocess.CalledProcessError(process.returncode, cmd)
    print("Command finished successfully.")


def get_encoder_flags(is_colab_gpu: bool) -> str:
    """Returns the best encoder flags based on hardware."""
    if is_colab_gpu:
        return "-c:v h264_nvenc -preset p4 -cq 23 -b:v 6M -c:a aac -b:a 192k"
    elif platform.system() == "Darwin":
        return "-c:v h264_videotoolbox -b:v 6M -c:a aac -b:a 192k"
    else:
        return "-c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -c:a aac -b:a 192k"


def build_input_string(paths: list, start_time: float) -> str:
    """
    Constructs the ffmpeg input string.
    Applies -ss ONLY to the first file to prevent data loss in split files.
    """
    inputs = []
    for i, p in enumerate(paths):
        if i == 0:
            inputs.append(f'-ss {start_time} -i "{p}"')
        else:
            inputs.append(f'-i "{p}"')
    return " ".join(inputs)


# ==========================================
# Verification render (10s preview)
# ==========================================
def render_verification_preview(p_side, p_front, p_360, aL, aR, out_path,
                                  ss_side, ss_front, ss_360, is_gpu: bool = False) -> None:
    vid_front = f'-ss {ss_front} -i "{p_front[0]}"'
    vid_side = f'-ss {ss_side} -i "{p_side[0]}"'
    vid_360 = f'-ss {ss_360} -i "{p_360[0]}"'

    # Audio from side camera (reference)
    aud_L = f'-ss {ss_side} -i "{aL[0]}"'
    aud_R = f'-ss {ss_side} -i "{aR[0]}"'

    filters = (
        "[0:v]scale=-2:360,fps=24,format=yuv420p[vF];"
        "[1:v]scale=-2:360,fps=24,format=yuv420p[vS];"
        "[2:v]scale=-2:360,fps=24,format=yuv420p[v3];"
        "[vF][vS][v3]hstack=inputs=3[v_out];"
        "[3:a][4:a]join=inputs=2:channel_layout=stereo[a_out]"
    )

    # CPU 'ultrafast' for preview regardless of is_gpu (safest for a quick check)
    enc_flags = "-c:v libx264 -preset ultrafast -crf 23"

    cmd = (
        f'ffmpeg -y {vid_front} {vid_side} {vid_360} {aud_L} {aud_R} '
        f'-filter_complex "{filters}" -map "[v_out]" -map "[a_out]" '
        f'{enc_flags} -t 10 "{out_path}" -hide_banner -stats'
    )
    _run_cmd(cmd)


# ==========================================
# Per-camera final renders
# ==========================================
def render_side_camera(video_paths, aL_paths, aR_paths, output_path, start_time,
                        is_colab_gpu: bool = False) -> None:
    print("\n   Configuring SIDE camera render...")

    vid_in = build_input_string(video_paths, start_time)
    aL_in = build_input_string(aL_paths, start_time)
    aR_in = build_input_string(aR_paths, start_time)

    n_parts = len(video_paths)
    n_audio = len(aL_paths)

    v_cat = "".join([f"[{i}:v]" for i in range(n_parts)]) + f"concat=n={n_parts}:v=1:a=0[v_cat_raw];"
    aL_cat = "".join([f"[{i + n_parts}:a]" for i in range(n_audio)]) + f"concat=n={n_audio}:v=0:a=1[aL];"
    aR_cat = "".join([f"[{i + n_parts + n_audio}:a]" for i in range(n_audio)]) + f"concat=n={n_audio}:v=0:a=1[aR];"

    full_filter = (
        f"{v_cat} {aL_cat} {aR_cat} [v_cat_raw]fps=24,format=yuv420p[vf]; "
        f"[aL][aR]join=inputs=2:channel_layout=stereo[a_out]"
    )

    cmd = (
        f'ffmpeg -y {vid_in} {aL_in} {aR_in} '
        f'-filter_complex "{full_filter}" -map "[vf]" -map "[a_out]" '
        f'{get_encoder_flags(is_colab_gpu)} "{output_path}" -hide_banner -stats'
    )
    _run_cmd(cmd)


def render_front_camera(video_paths, aL_paths, aR_paths, output_path, vid_start, aud_start,
                         is_colab_gpu: bool = False) -> None:
    print("\n   Configuring FRONT camera render...")

    vid_in = build_input_string(video_paths, vid_start)
    aL_in = build_input_string(aL_paths, aud_start)
    aR_in = build_input_string(aR_paths, aud_start)

    n_parts = len(video_paths)
    n_audio = len(aL_paths)

    v_cat = "".join([f"[{i}:v]" for i in range(n_parts)]) + f"concat=n={n_parts}:v=1:a=0[v_cat_raw];"
    aL_cat = "".join([f"[{i + n_parts}:a]" for i in range(n_audio)]) + f"concat=n={n_audio}:v=0:a=1[aL];"
    aR_cat = "".join([f"[{i + n_parts + n_audio}:a]" for i in range(n_audio)]) + f"concat=n={n_audio}:v=0:a=1[aR];"

    full_filter = (
        f"{v_cat} {aL_cat} {aR_cat} [v_cat_raw]fps=24,format=yuv420p[vf]; "
        f"[aL][aR]join=inputs=2:channel_layout=stereo[a_out]"
    )

    cmd = (
        f'ffmpeg -y {vid_in} {aL_in} {aR_in} '
        f'-filter_complex "{full_filter}" -map "[vf]" -map "[a_out]" '
        f'{get_encoder_flags(is_colab_gpu)} -shortest "{output_path}" -hide_banner -stats'
    )
    _run_cmd(cmd)


def render_360_camera(video_path_single, aL_paths, aR_paths, output_path, vid_start, aud_start,
                       is_colab_gpu: bool = False) -> None:
    print("\n   Configuring 360 camera render...")

    # Single file source, so -ss here is correct and necessary.
    vid_in = f'-ss {vid_start} -i "{video_path_single}"'
    aL_in = build_input_string(aL_paths, aud_start)
    aR_in = build_input_string(aR_paths, aud_start)

    n_audio = len(aL_paths)

    aL_cat = "".join([f"[{i + 1}:a]" for i in range(n_audio)]) + f"concat=n={n_audio}:v=0:a=1[aL];"
    aR_cat = "".join([f"[{i + 1 + n_audio}:a]" for i in range(n_audio)]) + f"concat=n={n_audio}:v=0:a=1[aR];"

    full_filter = (
        f"{aL_cat} {aR_cat} [0:v]fps=24,format=yuv420p[vf]; "
        f"[aL][aR]join=inputs=2:channel_layout=stereo[a_out]"
    )

    cmd = (
        f'ffmpeg -y {vid_in} {aL_in} {aR_in} '
        f'-filter_complex "{full_filter}" -map "[vf]" -map "[a_out]" '
        f'{get_encoder_flags(is_colab_gpu)} -shortest "{output_path}" -hide_banner -stats'
    )
    _run_cmd(cmd)
